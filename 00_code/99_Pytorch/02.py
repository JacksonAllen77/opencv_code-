import torch

a=torch.rand(2,3) # 生成一个2*3均匀分布的随机数矩阵
b=torch.randn(2,3) # 生成一个2*3正态分布的随机数矩阵
c=torch.normal(2,3,(2,3)) # 生成一个2*3的指定均值和标准差的正态分布
d=torch.distributions.Exponential(1).sample((2,3)) # 生成一个2*3的指数为1的指数分布
e=torch.distributions.Binomial(10,0.5).sample((2,3))  # 生成一个2*3的实验次数为10次，成功概率为0.5的二项分布
f=torch.distributions.Gamma(2,1).sample((2,3)) # 生成一个2*3的shape为2，scale为1的伽马分布
print(a)
print(b)
print(c)
print(d)
print(e)
print(f)