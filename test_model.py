import torch
import faiss
import numpy as np
import os
from tqdm import tqdm
import time

from utils import read_fbin
from model import MLP


def run_full_evaluation():
    # --- 1. 基本配置 ---
    QUERY_DATASET_PATH = 'data/query.public.100K.fbin'
    DATABASE_PATH = 'data/base.1M.fbin'
    MODEL_PATH = 'best_model.pth'
    DIM = 200
    K = 100
    # BATCH_SIZE 仅在 GPU 显存不足 (OOM) 时的 Fallback 模式下使用
    BATCH_SIZE = 512      
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 您可以把这里改成 "HNSW" 来测试 HNSW
    INDEX_TYPE = "HNSW"  
    
    print(f"模型使用设备(仅映射): {DEVICE}")
    print(f"测试索引类型: {INDEX_TYPE}")

    # --- 2. 加载数据 和 构建【CPU】索引 ---
    print("正在加载数据并构建 Faiss CPU 索引...")
    if not all(os.path.exists(p) for p in [QUERY_DATASET_PATH, DATABASE_PATH, MODEL_PATH]):
        print(f"错误: 缺少必要文件。请确保查询、数据库和模型文件都存在。")
        return

    Q_numpy = read_fbin(QUERY_DATASET_PATH).astype('float32')  # [N, D] 在 CPU
    X_numpy = read_fbin(DATABASE_PATH).astype('float32')      # [M, D] 在 CPU
    
    # 自动更新维度
    if Q_numpy.shape[1] != DIM:
        DIM = Q_numpy.shape[1]
        print(f"代码已自动更新向量维度为: {DIM}")
        
    num_queries = Q_numpy.shape[0]

    # 1. 构建 Ground Truth 索引 (永远使用 IndexFlatL2)
    print("正在构建 Ground Truth 索引 (IndexFlatL2)...")
    index_flat = faiss.IndexFlatL2(DIM)
    index_flat.add(X_numpy)
    print("Ground Truth 索引构建完毕。")

    # 2. 构建您要测试的 ANN 索引 (IVF 或 HNSW)
    print(f"正在构建 {INDEX_TYPE} 索引...")
    if INDEX_TYPE == "IVF":
        nlist = 1000
        quantizer = faiss.IndexFlatL2(DIM)
        index_ann = faiss.IndexIVFFlat(quantizer, DIM, nlist)
        print("正在训练 (Trained) IVF 索引...")
        index_ann.train(X_numpy)
        index_ann.add(X_numpy)
        index_ann.nprobe = 10
    elif INDEX_TYPE == "HNSW":
        M = 32
        index_ann = faiss.IndexHNSWFlat(DIM, M)
        print("正在向 HNSW 索引添加数据 (构建图)...")
        index_ann.add(X_numpy)
        index_ann.hnsw.efSearch = 64
    else:
        print("警告: INDEX_TYPE 设置为 'FLAT', 将与 Ground Truth 索引相同。")
        index_ann = index_flat
    
    print(f"CPU {INDEX_TYPE} 索引构建完毕，包含 {index_ann.ntotal} 个向量。")


    # --- 3. 加载模型（模型映射用 GPU，其余都在 CPU 上） ---
    print(f"正在从 '{MODEL_PATH}' 加载模型...")
    model = MLP(input_dim=DIM).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("模型加载成功。")

    print(f"\n--- 开始在全部 {num_queries} 个查询上进行评估 ---")

    # --- a) Baseline: 计算 Ground Truth 和 ANN Baseline ---
    
    # 1. 计算 Ground Truth (GT)
    # 检查是否已保存 Ground Truth 结果
    GT_INDICES_PATH = f"ground_truth_k{K}.npy"
    gt_time = 0.0
    
    if os.path.exists(GT_INDICES_PATH):
        print(f"正在从 '{GT_INDICES_PATH}' 加载已保存的 Ground Truth 结果...")
        start_load_gt = time.time()
        gt_pre_indices_all = np.load(GT_INDICES_PATH)
        end_load_gt = time.time()
        gt_time = end_load_gt - start_load_gt
        print(f"Ground Truth: 加载耗时 {gt_time:.2f} 秒")
    else:
        print(f"正在计算所有原始查询的KNN (Ground Truth, CPU, IndexFlatL2, K={K})... (这将很慢)")
        start_time_gt = time.time()
        _, gt_pre_indices_all = index_flat.search(Q_numpy, K)  
        end_time_gt = time.time()
        gt_time = end_time_gt - start_time_gt
        print(f"Ground Truth: 搜索 {num_queries} 个查询耗时 {gt_time:.2f} 秒")
        
        print(f"正在将 Ground Truth 结果保存到 '{GT_INDICES_PATH}' 以备将来使用...")
        np.save(GT_INDICES_PATH, gt_pre_indices_all)
        print("保存完毕。")

    # 2. 计算 ANN Baseline (用于对比速度和标准ANN召回率)
    print(f"正在计算所有原始查询的KNN (ANN Baseline, CPU, {INDEX_TYPE}, K={K})...")
    start_time_baseline = time.time()
    _, ann_pre_indices_all = index_ann.search(Q_numpy, K)  
    end_time_baseline = time.time()
    baseline_time = end_time_baseline - start_time_baseline
    print(f"ANN Baseline: 搜索 {num_queries} 个查询耗时 {baseline_time:.2f} 秒")


    # --- b) 【代码修改部分】Our Method: 先用 GPU 做映射，再回 CPU 搜 KNN ---
    print(f"正在计算所有映射后查询的KNN (Our Method: 映射用GPU, 搜索用CPU, {INDEX_TYPE}, K={K})...")

    model_inference_time = 0.0    # 累计模型推理时间（GPU）
    faiss_search_time = 0.0     # 累计 Faiss 搜索时间（CPU）
    q_mapped_all_numpy = np.zeros_like(Q_numpy, dtype='float32') # 预分配空间

    try:
        # 【尝试1】一次性完成所有查询的映射
        print(f"正在尝试使用 {DEVICE} 一次性映射所有 {num_queries} 个查询...")
        with torch.no_grad():
            start_model = time.time()
            # 尝试一次性加载所有数据到GPU
            Q_torch_all = torch.from_numpy(Q_numpy).to(DEVICE)
            # 尝试一次性推理
            Q_mapped_torch_all = model(Q_torch_all)
            if DEVICE == 'cuda':
                torch.cuda.synchronize()
            end_model = time.time()
            model_inference_time = end_model - start_model
            
            print(f"模型推理完成，耗时: {model_inference_time:.2f} 秒。正在将数据传回 CPU...")
            # 将结果传回 CPU
            q_mapped_all_numpy = Q_mapped_torch_all.detach().cpu().numpy().astype('float32')
            
            # 释放GPU显存
            del Q_torch_all
            del Q_mapped_torch_all
            if DEVICE == 'cuda':
                torch.cuda.empty_cache()

    except (torch.cuda.OutOfMemoryError, RuntimeError):
        # 【尝试2: Fallback】如果一次性映射失败 (OOM)，则退回 (Fallback) 到 BATCH 模式进行映射
        # 但仍然只在最后进行一次搜索
        print(f"警告: 一次性映射 {num_queries} 个查询导致 GPU 显存不足 (OOM) 或错误。")
        print(f"将自动切换回 Batch 模式 (BATCH_SIZE={BATCH_SIZE}) 进行映射...")
        model_inference_time = 0.0 # 重置计时器
        
        # 结果将收集在列表中，最后合并
        mapped_batches_list = []
        
        with torch.no_grad():
            for i in tqdm(range(0, num_queries, BATCH_SIZE), desc="Batch 映射中..."):
                q_batch_numpy = Q_numpy[i : i + BATCH_SIZE]
                
                start_model_batch = time.time()
                q_batch_torch = torch.from_numpy(q_batch_numpy).to(DEVICE)
                q_mapped_batch_torch = model(q_batch_torch)
                if DEVICE == 'cuda':
                    torch.cuda.synchronize()
                end_model_batch = time.time()
                model_inference_time += (end_model_batch - start_model_batch)
                
                # 收集 CPU 上的结果
                mapped_batches_list.append(q_mapped_batch_torch.detach().cpu().numpy())
                
                # 及时释放显存
                del q_batch_torch
                del q_mapped_batch_torch
                if DEVICE == 'cuda':
                    torch.cuda.empty_cache()
        
        print("所有 Batch 映射完成。正在合并结果...")
        # 合并所有批次的结果
        q_mapped_all_numpy = np.concatenate(mapped_batches_list, axis=0).astype('float32')
        print(f"Batch 映射总耗时: {model_inference_time:.2f} 秒")
        # --- Fallback 结束 ---

    #一次性完成所有映射后查询的搜索
    print(f"正在使用 CPU {INDEX_TYPE} 索引一次性搜索所有 {num_queries} 个映射后的查询...")
    start_knn = time.time()
    _, ann_post_indices_all = index_ann.search(q_mapped_all_numpy, K)  
    end_knn = time.time()
    faiss_search_time = end_knn - start_knn
    print(f"Faiss 搜索完成，耗时: {faiss_search_time:.2f} 秒")

    total_time = model_inference_time + faiss_search_time

    # --- c) 计算两种召回率 (与 Ground Truth 对比) ---
    print("正在计算最终的平均召回率 Recall@K...")

    total_ann_recall = 0.0  # ANN Baseline vs GT
    total_model_recall = 0.0 # Our Method vs GT

    for i in range(num_queries):
        # Ground Truth 邻居
        gt_neighbors = set(gt_pre_indices_all[i])
        
        # 1. ANN Baseline 的邻居
        ann_neighbors = set(ann_pre_indices_all[i])
        
        # 2. Our Method 的邻居
        model_neighbors = set(ann_post_indices_all[i])

        # ANN Baseline vs GT
        total_ann_recall += len(gt_neighbors.intersection(ann_neighbors)) / K

        # Our Method vs GT
        total_model_recall += len(gt_neighbors.intersection(model_neighbors)) / K


    average_ann_recall = total_ann_recall / num_queries
    average_model_recall = total_model_recall / num_queries

    print("\n--- 评估完成 ---")
    print(f"K = {K}")
    print(f"Ground Truth (FlatL2) 搜索总耗时: {gt_time:.2f} 秒")
    print(f"ANN Baseline ({INDEX_TYPE}) 搜索总耗时: {baseline_time:.2f} 秒")
    print(f"Our Method (GPU映射 + CPU {INDEX_TYPE}) 总耗时: {total_time:.2f} 秒")
    print(f"  - 模型推理总耗时 (GPU): {model_inference_time:.2f} 秒")
    print(f"  - KNN 搜索总耗时 (CPU, {INDEX_TYPE}): {faiss_search_time:.2f} 秒")
    
    print("\n--- 召回率 (与 Ground Truth 对比) ---")
    print(f"映射前原 ANN 召回率 (Recall@{K}): {average_ann_recall:.4%}")
    print(f"映射后召回率 (Recall@{K}): {average_model_recall:.4%}")


if __name__ == '__main__':
    run_full_evaluation()