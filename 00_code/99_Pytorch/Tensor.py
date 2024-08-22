import torch
from torch import tensor
# Scalar——标量【单一数值】
a=tensor(42.)
# Vector——向量【一组数值】
b=tensor([1.2,5.5,3.9])
# Matrix——矩阵【二维数组】例：图像数据（灰度图）
c= torch.tensor([[1.0, 2.0], [3.0, 4.0]])
# 3D Tensor【三位数据】例：图像数据（RGB图）
d=torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
