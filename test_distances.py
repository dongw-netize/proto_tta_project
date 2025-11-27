import torch
import numpy as np
import os

# 导入你项目中的 .py 文件
from utils import read_fbin  # <--- 修改在这里
from model import MLP  # <--- 修改在这里

# --- 1. 基本配置 ---
QUERY_DATASET_PATH = "data/query.public.100K.fbin"
MODEL_PATH = "best_model.pth"
DIM = 200  # 将被自动更新
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_average_l2_distance():
    # --- 2. 加载数据 ---
    print(f"使用设备: {DEVICE}")
    print(f"正在加载查询数据: {QUERY_DATASET_PATH}")
    Q_numpy = read_fbin(QUERY_DATASET_PATH).astype("float32")

    # 自动更新维度
    global DIM
    if Q_numpy.shape[1] != DIM:
        DIM = Q_numpy.shape[1]
        print(f"代码已自动更新向量维度为: {DIM}")

    # --- 3. 加载模型 ---
    print(f"正在从 '{MODEL_PATH}' 加载模型...")
    model = MLP(input_dim=DIM).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("模型加载成功。")

    # --- 4. 执行映射并计算 ---
    print("正在计算平均L2距离...")

    # 将 NumPy 数据转换为 GPU 上的 Torch Tensor
    Q_torch = torch.from_numpy(Q_numpy).to(DEVICE)

    # 执行模型推理 (不计算梯度)
    with torch.no_grad():
        T_Q_torch = model(Q_torch)

    # --- 5. 计算 Q 和 T(Q) 之间的平均 L2 距离 ---
    avg_distance = torch.norm(Q_torch - T_Q_torch, p=2, dim=1).mean()

    print("\n--- 评估完成 ---")
    # .item() 将0维张量转换为一个 Python 数字
    print(f"映射前 (Q) 与映射后 (T(Q)) 之间的平均L2距离: {avg_distance.item():.4f}")


if __name__ == "__main__":
    get_average_l2_distance()
