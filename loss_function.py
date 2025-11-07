# loss_function.py

import torch
import torch.nn as nn
import torch.nn.functional as F  # 确保 F 已导入
import numpy as np
# import ot # <--- 1. 已移除 OT 库
import warnings
from utils import differentiable_matrix_sqrt
import faiss 

warnings.filterwarnings("ignore", category=UserWarning, message="Input sums are not equal up to tolerance.*")

class CustomLoss(nn.Module):
    def __init__(self, X_data, Q_data_pre_computed, alpha, beta, lamb, K, tau, epsilon, delta, device='cpu'):
        """
        初始化总损失函数。
        【注意】: epsilon 参数现在已不再使用 (因为KL散度不需要它)，您可以稍后从 main.py 中移除它。
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
        self.q_pre_computed = Q_data_pre_computed

    #对应了pdf中最后总损失：λΩ(θ) 惩罚项
    def _l2_regularization(self, model):
        """计算模型参数的L2正则化项。"""
        reg_loss = 0.0
        for param in model.parameters():
            reg_loss += torch.sum(param.pow(2))
        return reg_loss / 2.0
    
    # KL散度
    def _loss_knn_consistency(self, T_q_batch, q_indices, faiss_index):

        batch_size = T_q_batch.shape[0] #批次的大小
        # 1. 搜索映射后的K-NN (与之前相同)
        # --- 这是您要的 CPU 搜索版本 ---
        q_search_numpy = T_q_batch.detach().cpu().numpy()
        if q_search_numpy.dtype != 'float32':
            q_search_numpy = q_search_numpy.astype('float32')
        _, post_indices_np = faiss_index.search(q_search_numpy, self.K)
        post_indices = torch.from_numpy(post_indices_np).to(self.device)
        # --- ------------------------ ---
        
        # 2. 计算映射后的权重 (与之前相同)
        X_neighbors = self.X[post_indices]
        l2_dist_sq = ((T_q_batch.unsqueeze(1) - X_neighbors)**2).sum(dim=-1)
        post_weights_batch = torch.softmax(-l2_dist_sq / self.tau, dim=1)
        
        batch_knn_loss = 0.0
        for i in range(batch_size):
            # 3. 获取映射前的 K-NN 数据 (与之前相同)
            original_q_index = q_indices[i].item()
            pre_indices_np = self.q_pre_computed['indices'][original_q_index] 
            pre_weights = self.q_pre_computed['weights'][original_q_index] # 映射前的权重 (Tensor)
            pre_indices = torch.from_numpy(pre_indices_np).long().to(self.device) # 映射前的索引 (Tensor)
            
            # 4. 获取当前查询的映射后 K-NN 数据
            post_weights_i = post_weights_batch[i] # 映射后的权重
            post_indices_i = post_indices[i] # 映射后的索引

            # 5. 在 pre 和 post 邻居的“并集”上构建两个概率分布
            union_indices = torch.unique(torch.cat([pre_indices, post_indices_i])) 
            
            # 创建一个从 "全局数据库索引" 到 "并集内索引" 的映射
            map_union = {idx.item(): j for j, idx in enumerate(union_indices)}
            
            # 在并集上创建两个空的概率分布
            p_m = torch.zeros(len(union_indices), device=self.device)       # 对应 P_m (映射前)
            q_m_theta = torch.zeros(len(union_indices), device=self.device) # 对应 Q_m(theta) (映射后)
            
            # 填充 p_m (映射前的权重)
            for j, idx in enumerate(pre_indices):
                p_m[map_union[idx.item()]] = pre_weights[j]
                
            # 填充 q_m_theta (映射后的权重)
            for j, idx in enumerate(post_indices_i):
                q_m_theta[map_union[idx.item()]] = post_weights_i[j]
            
            # 6.防止 log(0)
            p_m = torch.clamp(p_m, min=1e-8)
            p_m = p_m / p_m.sum()
            
            q_m_theta = torch.clamp(q_m_theta, min=1e-8)
            q_m_theta = q_m_theta / q_m_theta.sum()

            # 7.计算 KL 散度: KL(p_m || q_m_theta)
            # PyTorch 的 F.kl_div(input, target) 计算的是 sum(target * (log(target) - input))
            # 需要 input = log(q_m_theta), target = p_m
            kl_loss = F.kl_div(q_m_theta.log(), p_m, reduction='sum', log_target=False)
            
            batch_knn_loss += kl_loss

        return batch_knn_loss / batch_size
    

    def forward(self, model, q_batch, q_indices, faiss_index):
        """
        计算一个训练batch的总损失。它像一个总指挥，把前面所有独立的损失计算部分串联起来，完成一次完整的计算
        """
        # 1. 对当前批次的数据进行映射，这一步必须带梯度
        T_q_batch = model(q_batch)
        
        # 2. 【修改】使用 T_q_batch 计算分布对齐损失 (已注释掉)
        # loss_dist = self._loss_distribution_alignment(T_q_batch)
        loss_dist = torch.tensor(0.0) # 保持不变
        
        # 3. 使用 T_q_batch 计算KNN结构一致性损失 (现在使用KL散度)
        loss_knn = self._loss_knn_consistency(T_q_batch, q_indices, faiss_index)
        
        # 4. 计算L2正则化项
        loss_reg = self._l2_regularization(model)
        
        # 5. 加权求和得到总损失 (保持不变)
        # total_loss = self.alpha * loss_dist + self.beta * loss_knn + self.lamb * loss_reg
        total_loss = self.beta * loss_knn + self.lamb * loss_reg # 使用不含 loss_dist 的新公式
        
        return total_loss, loss_dist, loss_knn