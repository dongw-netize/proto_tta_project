import torch
import faiss
import numpy as np
import os
import ot
import matplotlib.pyplot as plt
from utils import read_fbin
from model import MLP
from loss_function import CustomLoss

#epoch add to 200
def main():
    # --- 1. 超参数设置 ---
    QUERY_DATASET_PATH = 'data/query.public.100K.fbin'
    DATABASE_PATH = 'data/base.1M.fbin'
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
    ALPHA = 0
    #控制 KNN结构一致性损失 loss_knn 的重要性
    BETA = 50
    #控制 L2正则化项 loss_reg 的强度。loss_reg 的目标: 惩罚模型中过大的参数值，防止模型变得过于复杂而产生过拟合 
    LAMBDA = 0.0001  #上次0.01
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {DEVICE}")
     # 【新增】定义检查点和最佳模型路径
    CHECKPOINT_DIR = "checkpoints"
    BEST_MODEL_PATH = "best_model.pth"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True) # 自动创建文件夹

    # --- 2. 数据加载与预处理 ---
    if not os.path.exists(QUERY_DATASET_PATH):
        print(f"错误: 查询数据集文件 '{QUERY_DATASET_PATH}' 未找到。请确保文件路径正确。")
        return
    if not os.path.exists(DATABASE_PATH):
        print(f"错误: 数据库文件 '{DATABASE_PATH}' 未找到。请确保文件路径正确。")
        return

    print("正在加载查询集和数据库...")
    Q_numpy = read_fbin(QUERY_DATASET_PATH)
    X_numpy = read_fbin(DATABASE_PATH)

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
    index = faiss.IndexFlatL2(DIM)
    if X_numpy.dtype != 'float32':
        X_numpy = X_numpy.astype('float32')
    index.add(X_numpy)
    print(f"Faiss索引 (L2 Distance) 构建完毕，包含 {index.ntotal} 个向量。")

    # --- 4. 预计算映射前的KNN分布 ---
    print("正在为原始查询集Q预计算KNN...")
    if Q_numpy.dtype != 'float32':
        Q_numpy = Q_numpy.astype('float32')
    #初始查询的相似度分数，和K个最近邻的数据库索引
    pre_scores, pre_indices = index.search(Q_numpy, K)
    pre_weights_unnorm = np.exp(-pre_scores / TAU) #距离越小权重越大，因此要在softmax前取负
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
    # 【新增】尝试从检查点加载，以恢复训练
    start_epoch = 0
    best_loss = float('inf') # 初始化最佳损失为无穷大
    latest_checkpoint_path = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pth")
    if os.path.exists(latest_checkpoint_path):
        print(f"正在从检查点 '{latest_checkpoint_path}' 恢复训练...")
        checkpoint = torch.load(latest_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']
        print(f"已恢复到 Epoch {start_epoch}。当前最佳损失为: {best_loss:.4f}")

    # ... (用于Matplotlib可视化的列表) ...
    batch_losses = []
    epoch_avg_losses = []

    # --- 7. 完整的训练循环 ---
    print("\n--- 开始完整的训练循环 ---")
    NUM_EPOCHS = 10  # 训练的总轮数
    BATCH_SIZE = 256  # 每个批次的大小

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train() # 设置为训练模式
        epoch_loss_sum = 0.0
        shuffled_indices = torch.randperm(Q.shape[0])
        # 【修改】在epoch循环的末尾打印，需要一个变量来存储最后一个batch的损失详情
        last_batch_details = {}
        
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
            if (i // BATCH_SIZE) % 100 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{i//BATCH_SIZE}], Loss: {total_loss.item():.4f}")
                loss_reg = loss_fn._l2_regularization(model) # 需要计算正则化项
                print(f"  - (加权) 分布对齐损失 L_dist: {loss_fn.alpha * loss_dist.item():.4f}")
                print(f"  - (加权) KNN一致性损失 L_kNN: {loss_fn.beta * loss_knn.item():.4f}")
                print(f"  - (加权) 正则化项 L_reg: {loss_fn.lamb * loss_reg.item():.4f}")

            # 存储最后一个batch的损失详情，用于epoch总结
            if i + BATCH_SIZE >= Q.shape[0]:
                loss_reg = loss_fn._l2_regularization(model)
                last_batch_details = {
                    'dist': loss_fn.alpha * loss_dist.item(),
                    'knn': loss_fn.beta * loss_knn.item(),
                    'reg': loss_fn.lamb * loss_reg.item()
                }

        avg_epoch_loss = epoch_loss_sum / (len(shuffled_indices) / BATCH_SIZE)
        epoch_avg_losses.append(avg_epoch_loss)
        
        # 【修改】在epoch总结中也加入详细的损失打印
        print(f"--- Epoch {epoch+1} 完成 --- 平均损失: {avg_epoch_loss:.4f} ---")
        print(f"  (基于最后一个批次) - (加权) 分布对齐损失 L_dist: {last_batch_details.get('dist', 0):.4f}")
        print(f"  (基于最后一个批次) - (加权) KNN一致性损失 L_kNN: {last_batch_details.get('knn', 0):.4f}")
        print(f"  (基于最后一个批次) - (加权) 正则化项 L_reg: {last_batch_details.get('reg', 0):.4f}")


        # 保存检查点和最佳模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_epoch_loss,
            'best_loss': best_loss, # 把最佳损失也存进去
        }, latest_checkpoint_path)
        print(f"已保存当前 Epoch {epoch+1} 的检查点。")

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"*** 新的最佳模型！损失降至 {best_loss:.4f}。已保存到 '{BEST_MODEL_PATH}' ***")

    print("\n--- 训练结束 ---")
    
    # ... (Matplotlib绘图部分不变) ...
    print("正在绘制损失曲线图...")
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_avg_losses, marker='o')
    plt.title("Per-Epoch Average Loss")
    plt.xlabel("Epoch Number")
    plt.ylabel("Average Loss")
    plt.grid(True)
    plt.savefig("loss_curve.png")
    print("损失曲线图已保存为 loss_curve.png")

if __name__ == '__main__':
    main()