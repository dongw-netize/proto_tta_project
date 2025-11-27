import torch
import faiss
import numpy as np
import os
import ot
import matplotlib.pyplot as plt
from utils import read_fbin
from model import MLP
from loss_function import CustomLoss


# epoch add to 200
def main():
    # --- 1. 超参数设置 ---
    QUERY_DATASET_PATH = "data/query.public.100K.fbin"
    DATABASE_PATH = "data/base.1M.fbin"
    DIM = 200
    K = 100

    # Softmax函数的温度系数 (建议保持 0.2 或 0.1)
    TAU = 0.2
    EPSILON = 0.05
    DELTA = 1e-3

    # --- 权重配置建议 ---
    # MMD weight (建议 1.0 - 5.0)
    ALPHA = 5.0

    # [新增] Anchor weight (建议 5.0，防止跑偏)
    GAMMA = 0.5

    # 控制 KNN结构一致性损失 loss_knn 的重要性 (建议 15.0 - 20.0)
    BETA = 15.0

    # 控制 L2正则化项 loss_reg 的强度 (建议保留，防止过拟合)
    LAMBDA = 0.0001

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {DEVICE}")
    # 定义检查点和最佳模型路径
    CHECKPOINT_DIR = "checkpoints"
    BEST_MODEL_PATH = "best_model.pth"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)  # 自动创建文件夹

    # --- 2. 数据加载与预处理 ---
    if not os.path.exists(QUERY_DATASET_PATH):
        print(f"错误: 查询数据集文件 '{QUERY_DATASET_PATH}' 未找到。")
        return
    if not os.path.exists(DATABASE_PATH):
        print(f"错误: 数据库文件 '{DATABASE_PATH}' 未找到。")
        return

    print("正在加载查询集和数据库...")
    Q_numpy = read_fbin(QUERY_DATASET_PATH)  # 查询集
    X_numpy = read_fbin(DATABASE_PATH)  # 数据集

    # numpy 数组转成 PyTorch Tensor
    if Q_numpy.shape[1] != DIM:
        DIM = Q_numpy.shape[1]
        print(f"代码已自动更新向量维度为: {DIM}")

    Q = torch.from_numpy(Q_numpy).to(DEVICE)
    X = torch.from_numpy(X_numpy).to(DEVICE)
    print(f"查询集 Q 的形状: {Q.shape}")
    print(f"数据库 X 的形状: {X.shape}")

    # --- 3. 构建Faiss索引 (GPU版本) ---
    print("正在为数据库X构建Faiss GPU索引...")

    res = faiss.StandardGpuResources()
    cpu_index = faiss.IndexFlatL2(DIM)
    index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

    if X_numpy.dtype != "float32":
        X_numpy = X_numpy.astype("float32")

    index.add(X_numpy)
    print(f"Faiss GPU索引 (L2 Distance) 构建完毕，包含 {index.ntotal} 个向量。")

    # --- 4. 预计算映射前的KNN分布 ---
    print("正在为原始查询集Q预计算KNN...")
    if Q_numpy.dtype != "float32":
        Q_numpy = Q_numpy.astype("float32")

    pre_scores, pre_indices = index.search(Q_numpy, K)
    pre_weights_unnorm = np.exp(-pre_scores / TAU)
    pre_weights = pre_weights_unnorm / pre_weights_unnorm.sum(axis=1, keepdims=True)
    q_data_pre_computed = {
        "indices": pre_indices,
        "weights": torch.from_numpy(pre_weights).to(DEVICE),
    }
    print("预计算完成。")

    # --- 5. 初始化模型和损失函数 ---
    model = MLP(input_dim=DIM).to(DEVICE)
    loss_fn = CustomLoss(
        X_data=X,
        Q_data_pre_computed=q_data_pre_computed,
        alpha=ALPHA,
        beta=BETA,
        lamb=LAMBDA,
        gamma=GAMMA,  # [新增] 传入 Gamma
        K=K,
        tau=TAU,
        epsilon=EPSILON,
        delta=DELTA,
        device=DEVICE,
    )

    # --- 6. 初始化优化器 ---
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    start_epoch = 0
    best_loss = float("inf")
    latest_checkpoint_path = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pth")
    if os.path.exists(latest_checkpoint_path):
        print(f"正在从检查点 '{latest_checkpoint_path}' 恢复训练...")
        checkpoint = torch.load(latest_checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["loss"]
        print(f"已恢复到 Epoch {start_epoch}。")

    batch_losses = []
    epoch_avg_losses = []

    # --- 7. 完整的训练循环 ---
    print("\n--- 开始完整的训练循环 ---")
    NUM_EPOCHS = 200
    BATCH_SIZE = 1024

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        epoch_loss_sum = 0.0
        shuffled_indices = torch.randperm(Q.shape[0])
        last_batch_details = {}

        for i in range(0, Q.shape[0], BATCH_SIZE):
            indices = shuffled_indices[i : i + BATCH_SIZE]
            q_batch = Q[indices]

            optimizer.zero_grad()

            # [修改] 接收 loss_anchor
            total_loss, loss_dist, loss_knn, loss_anchor = loss_fn(
                model=model, q_batch=q_batch, q_indices=indices, faiss_index=index
            )

            total_loss.backward()
            optimizer.step()
            epoch_loss_sum += total_loss.item()

            # 打印训练过程中的信息
            if (i // BATCH_SIZE) % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{i//BATCH_SIZE}], Total Loss: {total_loss.item():.4f}"
                )
                loss_reg = loss_fn._l2_regularization(model)
                print(f"  - L_dist (MMD): {loss_fn.alpha * loss_dist.item():.4f}")
                print(f"  - L_knn (KL):   {loss_fn.beta * loss_knn.item():.4f}")
                print(
                    f"  - L_anc (MSE):  {loss_fn.gamma * loss_anchor.item():.4f}"
                )  # [新增]
                print(f"  - L_reg (L2):   {loss_fn.lamb * loss_reg.item():.4f}")

            # 存储最后一个batch的损失详情
            if i + BATCH_SIZE >= Q.shape[0]:
                loss_reg = loss_fn._l2_regularization(model)
                last_batch_details = {
                    "dist": loss_fn.alpha * loss_dist.item(),
                    "knn": loss_fn.beta * loss_knn.item(),
                    "anchor": loss_fn.gamma * loss_anchor.item(),  # [新增]
                    "reg": loss_fn.lamb * loss_reg.item(),
                }

        avg_epoch_loss = epoch_loss_sum / (len(shuffled_indices) / BATCH_SIZE)
        epoch_avg_losses.append(avg_epoch_loss)

        # epoch总结
        print(f"--- Epoch {epoch+1} 完成 --- 平均损失: {avg_epoch_loss:.4f} ---")
        print(f"  Last Batch - L_dist: {last_batch_details.get('dist', 0):.4f}")
        print(f"  Last Batch - L_knn:  {last_batch_details.get('knn', 0):.4f}")
        print(
            f"  Last Batch - L_anc:  {last_batch_details.get('anchor', 0):.4f}"
        )  # [新增]

        # 保存检查点
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_epoch_loss,
                "best_loss": best_loss,
            },
            latest_checkpoint_path,
        )
        print(f"已保存当前 Epoch {epoch+1} 的检查点。")

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(
                f"*** 新的最佳模型！损失 {best_loss:.4f}。已保存到 '{BEST_MODEL_PATH}' ***"
            )

    print("\n--- 训练结束 ---")

    print("正在绘制损失曲线图...")
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_avg_losses, marker="o")
    plt.title("Per-Epoch Average Loss")
    plt.xlabel("Epoch Number")
    plt.ylabel("Average Loss")
    plt.grid(True)
    plt.savefig("loss_curve.png")
    print("损失曲线图已保存为 loss_curve.png")


if __name__ == "__main__":
    main()
