import torch

# 创建一个3x2的张量
a = torch.tensor([[1, 2], [3, 4], [5, 6]])
# 改变张量的形状
b=a.view(3,-1) # 将3x2的张量重塑为2x3的张量(-1表示在给定行的情况下，让其自己计算列)
c=a.reshape(-1,6) # 将3x2的张量重塑为1x6的张量(-1表示在给定行的情况下，让其自己计算列)
d=a.t() #转置

print(b.size(),b.dtype)
print(c.size(),c.dtype)
print(d.size(),d.dtype)