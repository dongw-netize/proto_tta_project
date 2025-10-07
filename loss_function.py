# loss_function.py

import torch
import torch.nn as nn
import numpy as np
import ot
import warnings
from utils import differentiable_matrix_sqrt
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
        self.epsilon = epsilon
        self.delta = delta
        self.device = device
        
        self.X = X_data.to(self.device)
        self.q_pre_computed = Q_data_pre_computed
        
        # 预先计算数据库X的均值和协方差矩阵 (这是固定的，可以预计算)
        self.mu_X = torch.mean(self.X, dim=0)
        cov_X = torch.cov(self.X.T)
        D = cov_X.size(0)
        ridge_X = (torch.trace(cov_X) / D) * self.delta
        self.Sigma_X = cov_X + ridge_X * torch.eye(D, device=self.device)
        self.Sigma_X = (self.Sigma_X + self.Sigma_X.T) / 2.0


    def _l2_regularization(self, model):
        """计算模型参数的L2正则化项。"""
        reg_loss = 0.0
        for param in model.parameters():
            reg_loss += torch.sum(param.pow(2))
        return reg_loss / 2.0

    def _loss_distribution_alignment(self, T_q_batch):
        """
        【修改核心】
        计算 L_dist (分布级对齐项)，但现在基于当前 mini-batch 进行估计。
        """
        # --- 使用当前批次来估计 mu_theta 和 Sigma_theta ---
        mu_theta = torch.mean(T_q_batch, dim=0)
        
        # 当 batch size > 1 时才计算协方差
        if T_q_batch.shape[0] > 1:
            cov_theta = torch.cov(T_q_batch.T)
        else:
            # 如果 batch size 为 1, 协方差为 0
            cov_theta = torch.zeros((T_q_batch.shape[1], T_q_batch.shape[1]), device=self.device)

        D = cov_theta.size(0)
        # 使用 max(self.delta, 1e-4) 防止 delta 过小
        ridge_theta = (torch.trace(cov_theta) / D) * max(self.delta, 1e-4) 
        Sigma_theta = cov_theta + ridge_theta * torch.eye(D, device=self.device)
        Sigma_theta = (Sigma_theta + Sigma_theta.T) / 2.0
        
        term_mean = torch.sum((mu_theta - self.mu_X).pow(2))
        sqrt_Sigma_theta = differentiable_matrix_sqrt(Sigma_theta)
        sqrt_term_matrix = sqrt_Sigma_theta @ self.Sigma_X @ sqrt_Sigma_theta
        sqrt_term_matrix = (sqrt_term_matrix + sqrt_term_matrix.T) / 2.0
        sqrt_term_processed = differentiable_matrix_sqrt(sqrt_term_matrix)
        term_cov = torch.trace(self.Sigma_X + Sigma_theta - 2 * sqrt_term_processed)
        
        return term_mean + torch.clamp(term_cov, min=0.0)

    def _loss_knn_consistency(self, T_q_batch, q_indices, faiss_index):
        """
        计算 L_kNN (KNN结构一致性项)，此部分逻辑保持不变。
        """
        batch_size = T_q_batch.shape[0]

        q_search_numpy = T_q_batch.detach().cpu().numpy()
        if q_search_numpy.dtype != 'float32':
            q_search_numpy = q_search_numpy.astype('float32')
        _, post_indices_np = faiss_index.search(q_search_numpy, self.K)
        post_indices = torch.from_numpy(post_indices_np).to(self.device)

        X_neighbors = self.X[post_indices]
        
        ip_scores = (T_q_batch.unsqueeze(1) * X_neighbors).sum(dim=-1)
        post_weights_batch = torch.softmax(ip_scores / self.tau, dim=1)
        
        batch_knn_loss = 0.0
        for i in range(batch_size):
            original_q_index = q_indices[i].item()
            pre_indices_np = self.q_pre_computed['indices'][original_q_index]
            pre_weights = self.q_pre_computed['weights'][original_q_index]
            post_weights = post_weights_batch[i]

            # === 【修改】对权重进行截断和重归一化，提高Sinkhorn稳定性 ===
            pre_weights = torch.clamp(pre_weights, min=1e-8)
            pre_weights = pre_weights / pre_weights.sum()
            post_weights = torch.clamp(post_weights, min=1e-8)
            post_weights = post_weights / post_weights.sum()
            # =======================================================

            pre_indices = torch.from_numpy(pre_indices_np).long().to(self.device)
            union_indices = torch.unique(torch.cat([pre_indices, post_indices[i]]))
            support_vectors = self.X[union_indices]
            # === 【修改】对代价矩阵进行归一化 ===
            cost_matrix = torch.cdist(support_vectors, support_vectors).pow(2)
            scale = cost_matrix.median()
            cost_matrix = cost_matrix / (scale + 1e-8)
            # ====================================

            map_union = {idx.item(): j for j, idx in enumerate(union_indices)}
            p_m = torch.zeros(len(union_indices), device=self.device)
            q_m_theta = torch.zeros(len(union_indices), device=self.device)
            
            for j, idx in enumerate(pre_indices):
                p_m[map_union[idx.item()]] = pre_weights[j]
            for j, idx in enumerate(post_indices[i]):
                q_m_theta[map_union[idx.item()]] = post_weights[j]
            
            transport_plan = ot.sinkhorn(p_m, q_m_theta, cost_matrix, self.epsilon)
            wasserstein_dist_sq = torch.sum(transport_plan * cost_matrix)
            batch_knn_loss += wasserstein_dist_sq

        return batch_knn_loss / batch_size

    def forward(self, model, q_batch, q_indices, faiss_index):
        """
        【修改核心】
        计算一个批次的加权总损失。不再接收 T_Q_full。
        """
        # 1. 对当前批次的数据进行映射，这一步必须带梯度
        T_q_batch = model(q_batch)
        
        # 2. 【修改】使用 T_q_batch 计算分布对齐损失
        loss_dist = self._loss_distribution_alignment(T_q_batch)
        
        # 3. 使用 T_q_batch 计算KNN结构一致性损失
        loss_knn = self._loss_knn_consistency(T_q_batch, q_indices, faiss_index)
        
        # 4. 计算L2正则化项
        loss_reg = self._l2_regularization(model)
        
        # 5. 加权求和得到总损失
        total_loss = self.alpha * loss_dist + self.beta * loss_knn + self.lamb * loss_reg
        
        return total_loss, loss_dist, loss_knn