# loss_function.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
import faiss

# 忽略特定的警告
warnings.filterwarnings(
    "ignore", category=UserWarning, message="Input sums are not equal up to tolerance.*"
)


class CustomLoss(nn.Module):
    def __init__(
        self,
        X_data,
        Q_data_pre_computed,
        alpha,
        beta,
        lamb,
        gamma,  # [新增] Anchor Loss 权重
        K,
        tau,
        epsilon,
        delta,
        device="cpu",
    ):
        """
        初始化总损失函数。
        """
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.lamb = lamb
        self.gamma = gamma  # [新增]
        self.K = K
        self.tau = tau
        self.epsilon = epsilon
        self.delta = delta
        self.device = device

        # 数据库 X (Base)
        self.X = X_data.to(self.device)
        self.num_database = self.X.shape[0]  # 记录数据库大小

        # 预先将所有 pre-computed 数据加载到 GPU
        self.q_pre_computed_weights = Q_data_pre_computed["weights"].to(self.device)
        self.q_pre_computed_indices = (
            torch.from_numpy(Q_data_pre_computed["indices"]).long().to(self.device)
        )

    def _l2_regularization(self, model):
        """计算模型参数的L2正则化项。"""
        reg_loss = 0.0
        for param in model.parameters():
            reg_loss += torch.sum(param.pow(2))
        return reg_loss / 2.0

    # --- RBF 核函数 (保持你文件中的版本) ---
    def _rbf_kernel(self, X, Y, sigma_sq=None):
        """
        计算 RBF 核矩阵 K(X, Y) = exp(- ||x - y||^2 / (2 * sigma^2))
        """
        # 1. 计算成对欧氏距离平方
        x_norm = (X**2).sum(1).view(-1, 1)
        y_norm = (Y**2).sum(1).view(1, -1)
        dist_sq = x_norm + y_norm - 2.0 * torch.mm(X, Y.t())
        dist_sq = torch.clamp(dist_sq, min=0)

        # 2. 中位数启发式
        if sigma_sq is None:
            median_dist = torch.median(dist_sq.detach())
            sigma_sq = median_dist / 2.0
            if sigma_sq < 1e-6:
                sigma_sq = torch.tensor(1.0, device=self.device)

        if not isinstance(sigma_sq, torch.Tensor):
            sigma_sq = torch.tensor(sigma_sq, device=self.device)

        # 3. 计算核矩阵
        gamma = 1.0 / (sigma_sq + 1e-8)
        K_xy = torch.exp(-gamma * dist_sq)

        return K_xy, sigma_sq

    # --- MMD Loss 计算 (保持你文件中的版本) ---
    def _loss_mmd(self, source, target):
        """
        计算 Maximum Mean Discrepancy (MMD)
        """
        combined = torch.cat([source, target], dim=0)
        _, sigma_sq = self._rbf_kernel(combined, combined, sigma_sq=None)

        K_xx, _ = self._rbf_kernel(source, source, sigma_sq=sigma_sq)
        K_yy, _ = self._rbf_kernel(target, target, sigma_sq=sigma_sq)
        K_xy, _ = self._rbf_kernel(source, target, sigma_sq=sigma_sq)

        loss = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
        return torch.clamp(loss, min=0.0)

    # --- KNN Consistency Loss (保持不变) ---
    def _loss_knn_consistency(self, T_q_batch, q_indices, faiss_index):
        batch_size = T_q_batch.shape[0]

        # 1. KNN 搜索
        T_q_batch_detached = T_q_batch.detach()
        if not T_q_batch_detached.is_contiguous():
            T_q_batch_detached = T_q_batch_detached.contiguous()

        try:
            post_D, post_I = faiss_index.search(T_q_batch_detached, self.K)
        except Exception as e:
            q_search_numpy = T_q_batch_detached.cpu().numpy().astype("float32")
            post_D, post_I = faiss_index.search(q_search_numpy, self.K)

        if isinstance(post_I, np.ndarray):
            post_indices = torch.from_numpy(post_I).long().to(self.device)
        else:
            post_indices = post_I.long().to(self.device)

        # 2. 计算映射后的权重
        X_neighbors = self.X[post_indices]
        l2_dist_sq = ((T_q_batch.unsqueeze(1) - X_neighbors) ** 2).sum(dim=-1)
        post_weights_batch = torch.softmax(-l2_dist_sq / self.tau, dim=1)

        # 3. 逐样本计算 KL
        batch_kl = T_q_batch.new_tensor(0.0)

        for i in range(batch_size):
            original_q_index = q_indices[i]
            pre_indices_i = self.q_pre_computed_indices[original_q_index]
            pre_weights_i = self.q_pre_computed_weights[original_q_index]

            post_indices_i = post_indices[i]
            post_weights_i = post_weights_batch[i]

            union_indices = torch.unique(
                torch.cat([pre_indices_i, post_indices_i], dim=0)
            )

            pre_compare = union_indices.unsqueeze(1) == pre_indices_i.unsqueeze(0)
            post_compare = union_indices.unsqueeze(1) == post_indices_i.unsqueeze(0)

            p_m = (pre_compare.float() @ pre_weights_i.unsqueeze(1)).squeeze(-1)
            q_m = (post_compare.float() @ post_weights_i.unsqueeze(1)).squeeze(-1)

            p_m = torch.clamp(p_m, min=1e-8)
            p_m = p_m / p_m.sum()
            q_m = torch.clamp(q_m, min=1e-8)
            q_m = q_m / q_m.sum()

            kl_i = F.kl_div(q_m.log(), p_m, reduction="sum", log_target=False)
            batch_kl += kl_i

        return batch_kl / batch_size

    def forward(self, model, q_batch, q_indices, faiss_index):
        """
        计算一个训练batch的总损失。
        """
        batch_size = q_batch.shape[0]

        # 1. 对当前批次的数据进行映射
        T_q_batch = model(q_batch)

        # 2. 计算 MMD 分布对齐损失 (L_dist)
        if self.alpha > 0:
            idx = torch.randint(0, self.num_database, (batch_size,), device=self.device)
            X_batch = self.X[idx]
            loss_dist = self._loss_mmd(T_q_batch, X_batch)
        else:
            loss_dist = torch.tensor(0.0, device=self.device)

        # 3. 计算KNN结构一致性损失
        loss_knn = self._loss_knn_consistency(T_q_batch, q_indices, faiss_index)

        # 4. 计算L2正则化项 (Model Weights)
        loss_reg = self._l2_regularization(model)

        # 5. [新增] 计算 Anchor Loss (别跑太远)
        # 计算 f(q) 和 q 之间的均方误差
        loss_anchor = torch.mean(torch.sum((T_q_batch - q_batch) ** 2, dim=1))

        # 6. 加权求和得到总损失
        total_loss = (
            self.alpha * loss_dist
            + self.beta * loss_knn
            + self.lamb * loss_reg
            + self.gamma * loss_anchor  # [新增]
        )

        # 返回总损失和分项损失
        return total_loss, loss_dist, loss_knn, loss_anchor
