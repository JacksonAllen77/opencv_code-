import torch

# 准备数据
x=torch.tensor(intput_feature,dtype=float)
y=torch.tensor(labels,dtype=float)
# 权重参数初始化
weight=torch.randn((12,128),dtype=float,requires_grad=True) # weight是输入层到隐藏层的权重矩阵
biases=torch.randn(128,dtype=float,requires_grad=True) # 隐藏层的偏置项
weight2=torch.randn((128,1),dtype=float,requires_grad=True) # 从隐藏层到输出层的权重矩阵
biases2=torch.randn(1,dtype=float,requires_grad=True) # 输出层的偏置项
learning_rate=0.001 # 学习率
losses=[] # 初始化一个空列表 losses 用于存储每次迭代的损失值。

# 执行前向传播
for i in range (1000):
    hidden=x.mm(weight2)+biases # 进行输入层到隐藏层的输出【矩阵乘法mm+偏置项biases】
    hidden=torch.relu(hidden) # 使用relu对隐层层结果进行非线性变化
    predictions=hidden.mm(weight2)+biases2 # 进行隐藏层到输入层的输出【矩阵乘法mm+偏置项biases】
    loss=torch.mean((predictions-y)**2) # 计算预测值与标签y之间的均方误差MSE损失
    losses.append(loss.data.numpy()) # 将上方计算的损失加入到列表中
    #打印损失值
    if i % 100==0:
        print("loss",loss)
    # 反向传播计算
    loss.backward() # 开始反向传播计算
    # 更新参数——用学习率乘以对应参数的梯度，并从原参数中减去，更新参数
    weight.data.add_(-learning_rate*weight.grad.data)
    biases.data.add(-learning_rate*biases.grad.data)
    weight2.data.add_(-learning_rate*weight2.grad.data)
    biases2.data.add(-learning_rate*biases2.grad.data)
    # 每次迭代将梯度清零
    weight.grad.data.zero_()
    biases.grad.data.zero_()
    weight2.grad.data.zero_()
    biases2.grad.data.zero_()

