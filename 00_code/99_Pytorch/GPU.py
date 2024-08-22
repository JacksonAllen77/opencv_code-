import torch
import torch.nn as nn # 用于构建神经网络所需的层和功能
import numpy as np
#准备训练数据
x_values=[i for i in range(11)] # x为从 0 到 10 的整数列表
x_train=np.array(x_values,dtype=np.float32).reshape(-1,1) # 利用Numpy构建数组，并将其转列向量
y_values=[2*i+1 for i in x_values] # 线性回归方程y = 2x + 1
y_train=np.array(y_values,dtype=np.float32).reshape(-1,1)
#定义线性回归模型
class  LinearRegressionModel(nn.Module): # 定义一个名为 LinearRegressionModel 的类，它继承了 nn.Module模块
    def __init__(self,input_dim,output_dim): # 构造函数定义该模型结构,input,output分表表示输入和输出的维度
        super(LinearRegressionModel,self).__init__() # 初始化父类 nn.Module组件
        self.linear=nn.Linear(input_dim,output_dim) # 定义了一个线性层，接收input_dim个输入并输出output_dim个输出
    def forward(self,x): # 定义前向传播方法
        out=self.linear(x) #输入数据x通过线性层生成输出out
        return out
# 实例化模型
input_dim = 1 # 定义输入的维度为1
output_dim = 1 # 定义输出的维度为1
model = LinearRegressionModel(input_dim, output_dim) # 使用定义的模型进行训练
# 使用GPU进行训练
device = torch.device('cuda'if torch.cuda.is_available() else'cpu')
model=model.to(device)
# 指定训练次数和损失函数
epochs=1000 # 指定迭代次数(训练次数)
learning_rate=0.01 # 指定学习率
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate) # 采用SGD优化器，优化所有模型参数
criterion=nn.MSELoss() # 使用均方误差MSE作为损失函数，横流昂预测值于真实值的误差
# 训练模型
for epoch in range(epochs):
    epoch += 1
    # 数据类型转换
    inputs=torch.from_numpy(x_train).to(device) # 将numpy的训练数据转换成tensor数据
    labels=torch.from_numpy(y_train).to(device)
    # 梯度每次迭代后要清零
    optimizer.zero_grad() # 每次迭代之前清零梯度，以避免累积
    # 前向传播
    outputs=model(inputs) # 传入输入，得到输出
    # 计算损失
    loss=criterion(outputs,labels) # 使用criterion计算预测值与预测值之间的误差
    # 反向传播
    loss.backward() #计算损失相对于模型参数的梯度
    # 更新权重参数
    optimizer.step() # 使用优化器更新模型参数（权重和偏置）
    if epoch %50 == 0:# 每隔50个迭代次数打印一次当前的训练损失
        print('epoch{},loss{}'.format(epoch,loss.item()))

# 测试模型预测结果
inputs = torch.from_numpy(x_train).to(device)  # 将测试数据转换为 PyTorch 张量并移动到 GPU 上
predict = model(inputs).cpu().detach().numpy()  # 通过模型进行预测，转回 CPU 并转换为 NumPy 数组print(predict)
print(predict)