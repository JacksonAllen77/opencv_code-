import torch

a=torch.tensor(5.23) # 标量（一个元素）
b=torch.tensor([1.24,2.55,3.54]) # 向量（一维张量）
c= torch.tensor([[1.0, 2.0], [3.0, 4.0]]) #矩阵（二维张量）
d=torch.zeros(3,4) # 生成一个3*4元素全为0的矩阵
e=torch.ones(4,3) # 生成一个3*4元素全为1的矩阵
f=torch.eye(3) # 生成一个3*3的单位矩阵
g=torch.arange(0,24,2).view(3,4) # 生成一个3*4的矩阵,根据起始值、结束值、步长
h=torch.randint(0, 100, (5, 3), dtype=torch.int8) # 生成一个 5×3 的在（0~100）的随机数矩阵
i=torch.rand(2,3) #生成一个2*3均匀分布的随机数矩阵
print(a)
print(b)
print(c)
print(d)
print(e)
print(f)
print(g)
print(h)
print(i)
# h中可以指定不同的数据类型，类似的还有[.float32,.float64,.float16,.int8,.uint8,.int16,.int32,.int64,.bool,
# .complex64,.complex128，.qint8，.quint8，.qint32]
