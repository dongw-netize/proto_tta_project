import torch
import faiss
import numpy as np
import os
from tqdm import tqdm # 引入一个漂亮的进度条库

from utils import read_fbin
from model import MLP

def run_full_evaluation():
    # --- 1. 基本配置 (与main.py保持一致) ---
    QUERY_DATASET_PATH = 'data/query.public.100K.fbin'
    DATABASE_PATH = 'data/base.1M.fbin'
    MODEL_PATH = 'proto_tta_model.pth'
    DIM = 200
    K = 10
    BATCH_SIZE = 512 # 分批处理查询，防止内存溢出
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"使用设备: {DEVICE}")

    # --- 2. 加载数据和Faiss索引 ---
    print("正在加载数据和构建Faiss索引...")
    if not all(os.path.exists(p) for p in [QUERY_DATASET_PATH, DATABASE_PATH, MODEL_PATH]):
        print(f"错误: 缺少必要文件。请确保查询、数据库和模型文件都存在。")
        return
        
    Q_numpy = read_fbin(QUERY_DATASET_PATH).astype('float32')
    X_numpy = read_fbin(DATABASE_PATH).astype('float32')
    
    DIM = Q_numpy.shape[1]

    index = faiss.IndexFlatL2(DIM)
    index.add(X_numpy)
    print(f"Faiss索引构建完毕，包含 {index.ntotal} 个向量。")

    # --- 3. 加载训练好的模型 ---
    print(f"正在从 '{MODEL_PATH}' 加载模型...")
    model = MLP(input_dim=DIM).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("模型加载成功。")

    # --- 4. 执行完整的评估 ---
    print(f"\n--- 开始在全部 {Q_numpy.shape[0]} 个查询上进行评估 ---")
    
    total_recall = 0.0
    num_queries = Q_numpy.shape[0]

    # a) 先一次性计算出所有原始查询的KNN结果
    print("正在计算所有原始查询的KNN...")
    _, pre_indices_all = index.search(Q_numpy, K)

    # b) 分批计算所有映射后查询的KNN结果
    print("正在分批计算所有映射后查询的KNN...")
    post_indices_all = np.zeros_like(pre_indices_all) # 创建一个空数组来存储结果

    with torch.no_grad(): #不需要计算梯度
        for i in tqdm(range(0, num_queries, BATCH_SIZE)): # 使用tqdm显示进度条
            q_batch_numpy = Q_numpy[i : i + BATCH_SIZE]
            q_batch_torch = torch.from_numpy(q_batch_numpy).to(DEVICE)
            print(q_batch_numpy[:10])
            q_mapped_batch_torch = model(q_batch_torch)  #模型接收一个批次的原始向量，并输出变换后的新向量
            print("前十个映射后：" )
            q_mapped_batch_numpy = q_mapped_batch_torch.cpu().numpy()
            print(q_mapped_batch_numpy[:10])
            _, post_indices_batch = index.search(q_mapped_batch_numpy, K)  #变换后的向量执行KNN
            post_indices_all[i : i + BATCH_SIZE] = post_indices_batch

    # c) 计算平均召回率
    print("正在计算最终的平均召回率...")
    for i in range(num_queries):
        pre_neighbors = set(pre_indices_all[i])
        post_neighbors = set(post_indices_all[i])
        #两个集合的交集的大小
        intersection_size = len(pre_neighbors.intersection(post_neighbors)) 
        total_recall += intersection_size / K #单个查询的召回率累加起来
        
    average_recall = total_recall / num_queries

    print("\n--- 评估完成 ---")
    print(f"在 {num_queries} 个查询上的平均召回率 (Recall@{K}): {average_recall:.4%}")


if __name__ == '__main__':
    # 如果想安装tqdm库，可以在终端运行: pip install tqdm
    run_full_evaluation()