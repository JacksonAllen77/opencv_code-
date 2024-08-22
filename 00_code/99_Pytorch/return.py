import  torch

x=torch.rand(1) # 创建一个
w=torch.rand(1,requires_grad=True)
b=torch.rand(1,requires_grad=True)
y=w*x
z=y+b
# 查看是否为模型需要梯度跟踪的张量
print(x.requires_grad,b.requires_grad,y.requires_grad,w.requires_grad)
# 查看是否为模型中叶子节点
print(x.is_leaf,b.is_leaf,y.is_leaf,w.is_leaf)
# 反向传播计算
z.backward(retain_graph=True) # 执行反向传播[如果不清空，梯度会累加起来]
print(w.grad) # 计算z对w的梯度
print(b.grad) # 计算z对b的梯度
