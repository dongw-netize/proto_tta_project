import torch
import faiss
import numpy as np
import os
import ot

from utils import read_fbin
from model import MLP
from loss_function import CustomLoss

def main():
    # --- 1. 超参数设置 ---
    DATASET_PATH = 'data/base.1M.fbin'  
    TOTAL_VECTORS = 1000000
    QUERY_SIZE = TOTAL_VECTORS // 10
    DATABASE_SIZE = TOTAL_VECTORS - QUERY_SIZE
    DIM = 200
    K = 30
    #Softmax函数的温度系数,较小的 TAU 值: 会让 Softmax 的输出变得“尖锐”。
    #相似度得分最高的那个邻居会获得接近 1 的权重，而其他邻居的权重会接近 0。这使得分布更接近于一个“硬”的、只选择最近邻的决策
    TAU = 0.3
    #EPSILON 值越大: 正则化效应越强。这使得最优传输计划 transport_plan 更“分散”，
    #计算过程更快、更稳定，但得到的是一个对真实Wasserstein距离的更模糊的近似。
    EPSILON = 0.05
    DELTA = 1e-3
    #控制 分布对齐损失 loss_dist 的重要性
    ALPHA = 1.0
    #控制 KNN结构一致性损失 loss_knn 的重要性
    BETA = 1.0
    #控制 L2正则化项 loss_reg 的强度。loss_reg 的目标: 惩罚模型中过大的参数值，防止模型变得过于复杂而产生过拟合 
    LAMBDA = 0.01
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {DEVICE}")

    # --- 2. 数据加载与预处理 ---
    if not os.path.exists(DATASET_PATH):
        print(f"错误: 数据集文件 '{DATASET_PATH}' 未找到。请确保数据集路径正确。")
        return

    print("正在从单个文件加载并拆分数据...")
    Q_numpy = read_fbin(DATASET_PATH, start_idx=0, chunk_size=QUERY_SIZE)
    X_numpy = read_fbin(DATASET_PATH, start_idx=QUERY_SIZE, chunk_size=DATABASE_SIZE)

    #numpy 数组转成 PyTorch Tensor(张量)，这样才能送进神经网络
    if Q_numpy.shape[1] != DIM:
        DIM = Q_numpy.shape[1]
        print(f"代码已自动更新向量维度为: {DIM}")

    Q = torch.from_numpy(Q_numpy).to(DEVICE)
    X = torch.from_numpy(X_numpy).to(DEVICE)
    print(f"查询集 Q 的形状: {Q.shape}")
    print(f"数据库 X 的形状: {X.shape}")

    # --- 3. 构建Faiss索引 ---
    print("正在为数据库X构建Faiss索引...")
    index = faiss.IndexFlatIP(DIM)
    if X_numpy.dtype != 'float32':
        X_numpy = X_numpy.astype('float32')
    index.add(X_numpy)
    print(f"Faiss索引 (Inner Product) 构建完毕，包含 {index.ntotal} 个向量。")

    # --- 4. 预计算映射前的KNN分布 ---
    print("正在为原始查询集Q预计算KNN...")
    if Q_numpy.dtype != 'float32':
        Q_numpy = Q_numpy.astype('float32')
    #初始查询的相似度分数，和K个最近邻的数据库索引
    pre_scores, pre_indices = index.search(Q_numpy, K)
    pre_weights_unnorm = np.exp(pre_scores / TAU)
    pre_weights = pre_weights_unnorm / pre_weights_unnorm.sum(axis=1, keepdims=True)
    q_data_pre_computed = {
        'indices': pre_indices,
        'weights': torch.from_numpy(pre_weights).to(DEVICE)
    }
    print("预计算完成。")

    # --- 5. 初始化模型和损失函数 ---
    model = MLP(input_dim=DIM).to(DEVICE)
    loss_fn = CustomLoss(
        X_data=X, Q_data_pre_computed=q_data_pre_computed,
        alpha=ALPHA, beta=BETA, lamb=LAMBDA,
        K=K, tau=TAU, epsilon=EPSILON, delta=DELTA, device=DEVICE
    )
    
 # --- 6. 初始化优化器 ---
    # 使用Adam优化器来更新模型参数，lr是学习率
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    # --- 7. 完整的训练循环 ---
    print("\n--- 开始完整的训练循环 ---")
    NUM_EPOCHS = 10  # 训练的总轮数
    BATCH_SIZE = 256  # 每个批次的大小

    for epoch in range(NUM_EPOCHS):
        # 在每个epoch开始时，打乱数据顺序以增加随机性
        shuffled_indices = torch.randperm(Q.shape[0])
        
        # 内层循环，遍历所有批次
        for i in range(0, Q.shape[0], BATCH_SIZE):
            # 1. 准备当前批次的数据
            indices = shuffled_indices[i : i + BATCH_SIZE]
            q_batch = Q[indices]
            
            # 2. 清空上一轮的梯度
            optimizer.zero_grad()
            
            # 3. 计算损失 (这部分就是你已经验证过的核心逻辑)
            total_loss, loss_dist, loss_knn = loss_fn(
                model=model, 
                q_batch=q_batch, 
                q_indices=indices,
                faiss_index=index
            )
            
            # 4. 反向传播，计算梯度
            total_loss.backward()
            
            # 5. 根据梯度，更新模型参数
            optimizer.step()

            # 打印训练过程中的信息
            if (i // BATCH_SIZE) % 100 == 0: # 每100个batch打印一次
                 print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{i//BATCH_SIZE}], Loss: {total_loss.item():.4f}")

        print(f"--- Epoch {epoch+1} 完成 ---")
        # 在每个epoch结束后，可以打印一次更详细的损失分项
        print(f"  - (加权) 分布对齐损失 L_dist: {loss_fn.alpha * loss_dist.item():.4f}")
        print(f"  - (加权) KNN一致性损失 L_kNN: {loss_fn.beta * loss_knn.item():.4f}")
        print(f"  - (加权) 正则化项 L_reg: {loss_fn.lamb * loss_fn._l2_regularization(model).item():.4f}")
    
    print("\n--- 训练结束 ---")

    # 训练结束后，你可以选择保存训练好的模型权重
    # torch.save(model.state_dict(), 'proto_tta_model.pth')
    # print("模型已保存到 proto_tta_model.pth")


if __name__ == '__main__':
    main()