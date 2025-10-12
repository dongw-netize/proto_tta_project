# utils.py

import numpy as np
import torch
import os

def read_fbin(filename, start_idx=0, chunk_size=None):
    """
    读取.fbin格式的向量文件。
    此版本修正为读取 [num_vectors, dim, data...] 格式的文件。
    """
    with open(filename, "rb") as f:
        # 读取文件头：向量总数和维度
        num_total, dim = np.fromfile(f, dtype="int32", count=2)
        
        # 文件头的长度是 2 * 4 = 8 字节
        header_size = 8
        
        # 定位到要读取的数据块的起始位置
        f.seek(header_size + start_idx * dim * 4) 
        
        if chunk_size is None:
            # 如果未指定块大小，则计算剩余的向量数
            remaining_bytes = os.fstat(f.fileno()).st_size - f.tell()
            num_vectors_to_read = remaining_bytes // (dim * 4)
        else:
            num_vectors_to_read = chunk_size
        
        # 计算要读取的浮点数总数
        count = num_vectors_to_read * dim
        data = np.fromfile(f, dtype="float32", count=count)
        
        return data.reshape(-1, dim)

#计算矩阵平方根 Σ 1/2
def differentiable_matrix_sqrt(A: torch.Tensor, eps: float = 1e-6, max_tries: int = 5):
    # 数值对称化
    A = 0.5 * (A + A.transpose(-1, -2))
    D = A.size(-1)
    # I = torch.eye(D, device=A.device, dtype=A.dtype) # I 未被使用

    # 使用 jitter 进行重试
    jitter_val = eps
    for i in range(max_tries):
        try:
            # 增加一个小的 jitter 来提高数值稳定性
            A_jittered = A + jitter_val * torch.eye(D, device=A.device, dtype=A.dtype)
            # 用双精度做分解更稳
            evals, evecs = torch.linalg.eigh(A_jittered.to(torch.float64))
            
            # 检查是否有负的特征值 (在 jitter 后理论上不应发生，但作为保险)
            if (evals < 0).any():
                # print(f"Warning: Negative eigenvalues found on try {i+1}. Clamping.")
                evals = torch.clamp(evals, min=1e-12)

            sqrt_evals = torch.sqrt(evals)
            
            # 将类型转换回来
            evecs = evecs.to(A.dtype)
            sqrt_evals = sqrt_evals.to(A.dtype)
            
            return (evecs * sqrt_evals.unsqueeze(-2)) @ evecs.transpose(-1, -2)
        except torch._C._LinAlgError: # 捕获 linalg 错误
            # 如果分解失败，增加 jitter 后重试
            jitter_val *= 10.0
            if i + 1 == max_tries:
                print("Matrix square root failed after max retries.")
    
    # 仍然失败：退化为仅对角近似（极少发生）
    print("Warning: eigh failed, falling back to diagonal approximation for sqrt.")
    diag = torch.diagonal(A, dim1=-2, dim2=-1)
    diag = torch.clamp(diag, min=1e-12)
    return torch.diag_embed(torch.sqrt(diag))
