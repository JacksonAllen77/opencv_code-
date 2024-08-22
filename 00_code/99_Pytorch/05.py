import torch

# 创建两个张量
a = torch.tensor([[1, 2], [3, 4],[5, 6]])
b = torch.tensor([[7, 8], [9, 10],[11,12]])

# 张量拼接
c=torch.vstack((a,b)) # 上下拼接
d=torch.hstack((a,b)) # 左右拼接
print(c)
print(d)
# 张量切片
e=a[0:2,0:2] #获取前2行，前2列的子张量
f=torch.split(a,2,dim=0) # 按行进行切割，每部分包含2行
g=torch.split(a,1,dim=1) # 按列进行切割，每部分包含1列
print(e)
print(f)
print(g)

# 张量索引
element=a[1,1] # 获取第2行第2列的元素
row=a[1,:] # 获取第2行的所有元素
column=a[:,1] # 获取第2列的所有元素
print(element)
print(row)
print(column)

