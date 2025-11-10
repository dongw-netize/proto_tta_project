# loss_function.py

import torch
import torch.nn as nn
import torch.nn.functional as F  # 确保 F 已导入
import numpy as np
# import ot # <--- 1. 已移除 OT 库
import warnings
from utils import differentiable_matrix_sqrt # <--- 确保这个导入存在
import faiss 

warnings.filterwarnings("ignore", category=UserWarning, message="Input sums are not equal up to tolerance.*")

class CustomLoss(nn.Module):
    def __init__(self, X_data, Q_data_pre_computed, alpha, beta, lamb, K, tau, epsilon, delta, device='cpu'):
        """
        初始化总损失函数。
        """
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.lamb = lamb
        self.K = K
        self.tau = tau
        self.epsilon = epsilon # <--- 此参数现在未使用
        self.delta = delta
        self.device = device
        
        self.X = X_data.to(self.device)

        # --- 【优化 1: 将预计算数据一次性移至GPU】 ---
        # 预先将所有 pre-computed 数据加载到 GPU
        self.q_pre_computed_weights = Q_data_pre_computed['weights'].to(self.device)
        self.q_pre_computed_indices = torch.from_numpy(Q_data_pre_computed['indices']).long().to(self.device)
        # --- 【优化 1 结束】 ---

        # --- 【预计算数据库X的统计数据】 ---
        print("正在预计算数据库X的均值和协方差...")
        self.mu_X = torch.mean(self.X, dim=0)
        
        # [cite_start]计算协方差矩阵 (根据PDF [cite: 97])
        X_centered = self.X - self.mu_X 
        self.cov_X = (X_centered.T @ X_centered) / self.X.shape[0] + \
                       self.delta * torch.eye(self.X.shape[1], device=self.device)
        
        # 预计算协方差的平方根 (用于 L_dist)
        self.cov_X_sqrt = differentiable_matrix_sqrt(self.cov_X)
        print("预计算完成。")
        # --- 【预计算结束】 ---

    #对应了pdf中最后总损失：λΩ(θ) 惩罚项 [cite: 153]
    def _l2_regularization(self, model):
        """计算模型参数的L2正则化项。"""
        reg_loss = 0.0
        for param in model.parameters():
            reg_loss += torch.sum(param.pow(2))
        return reg_loss / 2.0
    
    # 对应了pdf中 的分布级对齐项 L_dist [cite: 106]
    def _loss_distribution_alignment(self, T_q_batch):
        """
        计算映射后查询批次(T_q_batch)与整个数据库(X)之间的
        高斯-瓦瑟斯坦距离 L_dist。
        使用批次统计量 (mu_theta_batch, cov_theta_batch) 来估计
        T(Q) 的全局统计量 (mu_theta, cov_theta [cite: 100, 101])。
        """
        
        # 1. 计算批次的均值和协方差 (估计 mu_theta, cov_theta [cite: 100, 101])
        mu_theta_batch = torch.mean(T_q_batch, dim=0)
        
        T_q_batch_centered = T_q_batch - mu_theta_batch
        cov_theta_batch = (T_q_batch_centered.T @ T_q_batch_centered) / T_q_batch.shape[0] + \
                          self.delta * torch.eye(T_q_batch.shape[1], device=self.device)

        # 2. 计算 L_dist (根据PDF [cite: 106] 的公式)
        
        # 均值项: ||mu_theta - mu_X||_2^2
        loss_mean = torch.sum((mu_theta_batch - self.mu_X) ** 2)
        
        # 协方差项: Tr(Sigma_theta + Sigma_X - 2 * (Sigma_theta^1/2 * Sigma_X * Sigma_theta^1/2)^1/2)
        
        # 计算 Sigma_theta^1/2
        cov_theta_batch_sqrt = differentiable_matrix_sqrt(cov_theta_batch)
        
        # 计算 (...) 内的项: Sigma_theta^1/2 * Sigma_X * Sigma_theta^1/2
        term_to_sqrt = cov_theta_batch_sqrt @ self.cov_X @ cov_theta_batch_sqrt
        
        # 计算 (...) 的 1/2 次方
        sqrt_term = differentiable_matrix_sqrt(term_to_sqrt)
        
        # 计算迹 Tr(...)
        loss_cov = torch.trace(self.cov_X + cov_theta_batch - 2 * sqrt_term)
        
        # 【修复】确保 L_dist (即 W2^2) 永远不会小于 0
        total_dist_loss = torch.clamp(loss_mean + loss_cov, min=0.0)
        
        return total_dist_loss
    # --- 【方法结束】 ---

    
    # [cite_start]KL散度 (L_kNN 的一种实现) [cite: 115]
    def _loss_knn_consistency(self, T_q_batch, q_indices, faiss_index):

        batch_size = T_q_batch.shape[0]

        # 1. KNN 搜索：优先走 GPU-tensor 路线，失败再退回 numpy
        T_q_batch_detached = T_q_batch.detach()
        if not T_q_batch_detached.is_contiguous():
            T_q_batch_detached = T_q_batch_detached.contiguous()

        try:
            # 优先尝试：Faiss 直接吃 torch tensor
            post_D, post_I = faiss_index.search(T_q_batch_detached, self.K)
        except Exception as e:
            # 退回旧方案：cpu().numpy()
            q_search_numpy = T_q_batch_detached.cpu().numpy().astype('float32')
            post_D, post_I = faiss_index.search(q_search_numpy, self.K)

        # 兼容两种返回类型：torch.Tensor 或 numpy.ndarray
        if isinstance(post_I, np.ndarray):
            post_indices = torch.from_numpy(post_I).long().to(self.device)
        else:  # torch.Tensor
            post_indices = post_I.long().to(self.device)

        # [cite_start]2. 计算映射后的权重 [cite: 109, 128]
        X_neighbors = self.X[post_indices]  # [B, K, D]
        l2_dist_sq = ((T_q_batch.unsqueeze(1) - X_neighbors) ** 2).sum(dim=-1)  # [B, K]
        post_weights_batch = torch.softmax(-l2_dist_sq / self.tau, dim=1)        # [B, K]

        # 3. 逐样本计算 KL
        batch_kl = T_q_batch.new_tensor(0.0)

        for i in range(batch_size):
            # 3.1 映射前 KNN（全部在 GPU 上取）
            # --- 【关键性能修复：移除 .item()】 ---
            original_q_index = q_indices[i]  # [快] 只是一个 0 维 GPU 张量
            # --- 【修复结束】 ---
            
            pre_indices_i = self.q_pre_computed_indices[original_q_index]    # [K]
            pre_weights_i = self.q_pre_computed_weights[original_q_index]    # [K]

            # 3.2 映射后 KNN
            post_indices_i = post_indices[i]        # [K]
            post_weights_i = post_weights_batch[i]  # [K]

            # 3.3 联合支撑集 U
            union_indices = torch.unique(torch.cat([pre_indices_i, post_indices_i], dim=0))  # [U]

            # 3.4 用布尔矩阵 + matmul 填 p_m, q_m
            pre_compare  = (union_indices.unsqueeze(1) == pre_indices_i.unsqueeze(0))   # [U, K]
            post_compare = (union_indices.unsqueeze(1) == post_indices_i.unsqueeze(0))  # [U, K]

            p_m = (pre_compare.float()  @ pre_weights_i.unsqueeze(1)).squeeze(-1)   # [U]
            q_m = (post_compare.float() @ post_weights_i.unsqueeze(1)).squeeze(-1)  # [U]

            # 3.5 防止 log(0) + 归一化
            p_m = torch.clamp(p_m, min=1e-8); p_m = p_m / p_m.sum()
            q_m = torch.clamp(q_m, min=1e-8); q_m = q_m / q_m.sum()

            # [cite_start]3.6 KL(p || q) [cite: 115]
            kl_i = F.kl_div(q_m.log(), p_m, reduction='sum', log_target=False)
            batch_kl += kl_i

        return batch_kl / batch_size
    

    def forward(self, model, q_batch, q_indices, faiss_index):
        """
        计算一个训练batch的总损失。
        """
        # 1. 对当前批次的数据进行映射
        T_q_batch = model(q_batch)
        
        # 2. 计算分布对齐损失
        loss_dist = self._loss_distribution_alignment(T_q_batch)
        
        # 3. 计算KNN结构一致性损失
        loss_knn = self._loss_knn_consistency(T_q_batch, q_indices, faiss_index)
        
        # 4. 计算L2正则化项
        loss_reg = self._l2_regularization(model)
        
        # [cite_start]5. 加权求和得到总损失 [cite: 152]
        total_loss = self.alpha * loss_dist + self.beta * loss_knn + self.lamb * loss_reg
        
        # 返回总损失和分项损失（用于打印日志）
        return total_loss, loss_dist, loss_knn