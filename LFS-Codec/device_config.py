import torch

# 检查是否有可用 GPU
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")