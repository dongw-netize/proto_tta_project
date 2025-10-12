# loss_function.py

import torch
import torch.nn as nn
import numpy as np
import ot
import warnings
from utils import differentiable_matrix_sqrt
import faiss 
# 训练的目的是用准确的knn训练，这样映射后的尽可能接近
warnings.filterwarnings("ignore", category=UserWarning, message="Input sums are not equal up to tolerance.*")

class CustomLoss(nn.Module):
    def __init__(self, X_data, Q_data_pre_computed, alpha, beta, lamb, K, tau, epsilon, delta, device='cpu'):
        """
        初始化总损失函数。这里是整个数据库的
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
        
        # dim=0意思是对每一列求均值，每一行是一个样本（一个 embedding）；每一列是一个维度（一个特征） 对应μX
        self.mu_X = torch.mean(self.X, dim=0)
        #计算整个数据库的self.X的样本协方差矩阵对应(ΣX),欧式
        cov_X = torch.cov(self.X.T)
        #协方差矩阵 cov_X 的第 0 维长度取出来，记为 D
        D = cov_X.size(0)
        #这里对应的记号高斯近似的ΣX；给协方差矩阵加上trace-scaled 岭正则（ridge regularization），用来在数值上稳定矩阵运算
        ridge_X = (torch.trace(cov_X) / D) * self.delta
        self.Sigma_X = cov_X + ridge_X * torch.eye(D, device=self.device)
        self.Sigma_X = (self.Sigma_X + self.Sigma_X.T) / 2.0

    #对应了pdf中最后总损失：λΩ(θ) 惩罚项
    def _l2_regularization(self, model):
        """计算模型参数的L2正则化项。"""
        reg_loss = 0.0
        #PyTorch提供的一个方法，它可以遍历模型中所有需要学习的参数
        for param in model.parameters():
            reg_loss += torch.sum(param.pow(2))
        return reg_loss / 2.0
    
    #分布级对齐损失
    def _loss_distribution_alignment(self, T_q_batch):
        #对应pdf的 记号与高斯分布 ：μθ均值
        mu_theta = torch.mean(T_q_batch, dim=0)
        
        # 对应pdf的 记号与高斯近似：Σθ
        if T_q_batch.shape[0] > 1:
            cov_theta = torch.cov(T_q_batch.T)
        else:
            # 如果 batch size 为 1, 协方差为 0 (至少需要两个样本才可以计算)
            cov_theta = torch.zeros((T_q_batch.shape[1], T_q_batch.shape[1]), device=self.device)

        D = cov_theta.size(0)
        # 使用 max(self.delta, 1e-4) 防止 delta δ过小；对应pdf记号与高斯近似的 Σθ 公式中的 + δI
        ridge_theta = (torch.trace(cov_theta) / D) * max(self.delta, 1e-4) 
        Sigma_theta = cov_theta + ridge_theta * torch.eye(D, device=self.device)
        Sigma_theta = (Sigma_theta + Sigma_theta.T) / 2.0
        
        #计算二阶Wasserstein距离
        term_mean = torch.sum((mu_theta - self.mu_X).pow(2))
        sqrt_Sigma_theta = differentiable_matrix_sqrt(Sigma_theta)
        sqrt_term_matrix = sqrt_Sigma_theta @ self.Sigma_X @ sqrt_Sigma_theta #@是矩阵乘法的意思
        sqrt_term_matrix = (sqrt_term_matrix + sqrt_term_matrix.T) / 2.0
        sqrt_term_processed = differentiable_matrix_sqrt(sqrt_term_matrix)
        term_cov = torch.trace(self.Sigma_X + Sigma_theta - 2 * sqrt_term_processed)
        
        return term_mean + torch.clamp(term_cov, min=0.0)
    
    #KNN 分支 = 内积/余弦相似度（大=近）；分布对齐分支 = W2 距离 核心bug！！
    def _loss_knn_consistency(self, T_q_batch, q_indices, faiss_index):

        batch_size = T_q_batch.shape[0] #批次的大小256
        #将PyTorch张量 T_q_batch 的梯度信息剥离，再将它从GPU（如果存在转移到CPU，最后转换成一个NumPy数组，并赋值给 q_search_numpy 变量，以便后续给Faiss库使用
        q_search_numpy = T_q_batch.detach().cpu().numpy()
        if q_search_numpy.dtype != 'float32':
            q_search_numpy = q_search_numpy.astype('float32')
        _, post_indices_np = faiss_index.search(q_search_numpy, self.K)
        #post_indices 是一个形状为 (batch_size, K) 的tensor，里面存的是邻居在数据库 self.X 中的位置索引
        post_indices = torch.from_numpy(post_indices_np).to(self.device)
        # 利用 post_indices 中的索引，从总数据库 self.X 中取出对应的邻居向量
        X_neighbors = self.X[post_indices]
        #批处理的方式来计算每个查询向量与其K个邻居向量之间的内积
        l2_dist_sq = ((T_q_batch.unsqueeze(1) - X_neighbors)**2).sum(dim=-1)
        post_weights_batch = torch.softmax(-l2_dist_sq / self.tau, dim=1)
        
        batch_knn_loss = 0.0
        for i in range(batch_size):
            #代表当前查询在整个数据集中的唯一ID
            original_q_index = q_indices[i].item()
            pre_indices_np = self.q_pre_computed['indices'][original_q_index] #一个NumPy数组，包含了当前查询在映射前的K个最近邻的全局索引
            pre_weights = self.q_pre_computed['weights'][original_q_index]
            post_weights = post_weights_batch[i]

            # === 【修改】对权重进行截断和重归一化，提高Sinkhorn稳定性 ===
            pre_weights = torch.clamp(pre_weights, min=1e-8)
            pre_weights = pre_weights / pre_weights.sum()
            post_weights = torch.clamp(post_weights, min=1e-8)
            post_weights = post_weights / post_weights.sum()
            # =======================================================

            pre_indices = torch.from_numpy(pre_indices_np).long().to(self.device)
            union_indices = torch.unique(torch.cat([pre_indices, post_indices[i]])) #当前查询所有相关邻居（映射前+映射后）的并集。这就是计算Wasserstein距离的“支撑集”
            support_vectors = self.X[union_indices]
            # === 【修改】对代价矩阵进行归一化 ===
            cost_matrix = torch.cdist(support_vectors, support_vectors).pow(2)
            scale = cost_matrix.median()
            cost_matrix = cost_matrix / (scale + 1e-8) #代价矩阵 Dij；PDF度量一致性这
            # ====================================

            map_union = {idx.item(): j for j, idx in enumerate(union_indices)}
            p_m = torch.zeros(len(union_indices), device=self.device)
            q_m_theta = torch.zeros(len(union_indices), device=self.device)
            
            for j, idx in enumerate(pre_indices):
                p_m[map_union[idx.item()]] = pre_weights[j]
            for j, idx in enumerate(post_indices[i]):
                q_m_theta[map_union[idx.item()]] = post_weights[j]
            
            transport_plan = ot.sinkhorn(p_m, q_m_theta, cost_matrix, self.epsilon) #最优运输方案 πij

            wasserstein_dist_sq = torch.sum(transport_plan * cost_matrix)
            batch_knn_loss += wasserstein_dist_sq

        return batch_knn_loss / batch_size

    def forward(self, model, q_batch, q_indices, faiss_index):
        """
        计算一个训练batch的总损失。它像一个总指挥，把前面所有独立的损失计算部分串联起来，完成一次完整的计算
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