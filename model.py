# model.py

import torch.nn as nn

class MLP(nn.Module):
    """
    定义映射函数 T_theta 的神经网络结构。
    这是一个简单的多层感知机(MLP)，用于学习从原始查询空间到目标索引空间的非线性映射。
    """
    def __init__(self, input_dim, hidden_dim=512):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # 输出维度必须与输入维度相同，以保持向量空间的一致性
            nn.Linear(hidden_dim, input_dim) 
        )

    def forward(self, x):
        return self.model(x)