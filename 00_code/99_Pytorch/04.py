import torch

a=torch.ones(3,4,dtype=int) # 生成一个3*4元素全为0的矩阵
print(a)
b=torch.arange(0,24,2,dtype=int).view(3,4) # 生成一个3*4的矩阵,根据起始值、结束值、步长

# 张量加减乘除运算
result = a - b  # 元素减法
result = a * b  # 元素乘法
result = a / b  # 元素除法
result = a + b  # 元素加法
print(result)
# 矩阵相乘运算
c=b.view(4,3)
result=torch.matmul(a,c) # 矩阵乘法[要求两矩阵可以相乘，且数据类型一致]
print(result)
# 转置运算
d=c.t()
print(d)