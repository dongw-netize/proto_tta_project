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

def differentiable_matrix_sqrt(matrix):
    """
    PDF:分布级对齐两个高斯分布的闭式公式里，会出现两次“矩阵平方根
    计算正半定矩阵的矩阵平方根，且该操作是可微分的。
    原理是利用特征值分解: A = V @ diag(L) @ V.T 
    则 A^0.5 = V @ diag(L^0.5) @ V.T
    """
    # 协方差矩阵是对称的，可以使用 torch.linalg.eigh
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    # 对特征值进行clamp，防止因数值不稳定出现负数
    sqrt_eigenvalues = torch.sqrt(torch.clamp(eigenvalues, min=0))
    return eigenvectors @ torch.diag(sqrt_eigenvalues) @ eigenvectors.T
