#处理数据
import numpy as np
#天气样本数据读取
import pandas as pd
#处理时间数据
import datetime
#可视化工具
import matplotlib.pyplot as plt
#数据预处理工具
from sklearn import preprocessing
#神经网络搭建
import torch
import torch.optim as optim
#忽略警告消息
import warnings
warnings.filterwarnings("ignore")



#-----------------------------------1、导入天气温度数据样本----------------------------------
features = pd.read_csv('F:/Pycharm/opencv/opencv/code/Pytorch/temps0.csv')
features.head()
print('数据维度before：',features.shape)
years = features['year']
months = features['month']
days = features['day']

#把年月日转换为str类型，再转换为datetime类格式                                             zip（）函数用于并行遍历
dates = [str(int(year))+'-'+str(int(month))+'-'+str(int(day)) for year,month,day in zip(years,months,days)]
dates = [datetime.datetime.strptime(date,'%Y-%m-%d') for date in dates]

#----------------------------------2、数据可视化绘图------------------------------------
#指定绘图风格
plt.style.use('fivethirtyeight')
#绘制4个图（最大值、前天、昨天、朋友预测） ax1 ax2为第一行两个子图，ax3 ax4为第二行两个子图  底部日期标签旋转45°
fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2,ncols=2,figsize=(10,10))
fig.autofmt_xdate(rotation = 45)

#标签值  当天真实温度最高温度 横轴日期 竖轴气温
ax1.plot(dates,features['actual'])
ax1.set_xlabel('');ax1.set_ylabel('Temperature');ax1.set_title('Max Temp')

#昨天的最高温度值
ax2.plot(dates,features['temp_1'])
ax2.set_xlabel('');ax2.set_ylabel('Temparature');ax2.set_title('Previous Max Temp')

#前天的最高温度
ax3.plot(dates,features['temp_2'])
ax3.set_xlabel('Date');ax3.set_ylabel('Temparature');ax3.set_title('Two Days Prior Max Temp')

#朋友预测
ax4.plot(dates,features['friend'])
ax4.set_xlabel('Date');ax4.set_ylabel('Temparature');ax4.set_title('Friend Esitimate')

#调整子图内边距
plt.tight_layout(pad=2)
plt.show()

#--------------------------------------3、数据预处理---------------------------------
#独热编码
features = pd.get_dummies(features)
features.head(5)
#将特征值和标签分开
labels = np.array(features['actual'])
features = features.drop('actual',axis = 1)
feature_list = list(features.columns)
features = np.array(features)
print('数据维度new：',features.shape)

# 特征值的Z-score归一化  使其均值为0 方差为1
input_features = preprocessing.StandardScaler().fit_transform(features)

#--------------------------------------4、搭建网络模型---------------------------------
#构建网络模型
input_size = input_features.shape[1]
hidden_size = 128
output_size = 1
batch_size = 16
my_nn = torch.nn.Sequential(
    torch.nn.Linear(input_size,hidden_size),
    torch.nn.Sigmoid(),
    torch.nn.Linear(hidden_size,output_size),
)
cost = torch.nn.MSELoss(reduction = 'mean')
optimizer = torch.optim.Adam(my_nn.parameters(),lr=0.001)

#训练网络
losses = []
for i in range(1000):
    batch_loss = []
    #使用MINI-Batch方法来训练
    for start in range(0,len(input_features),batch_size):
        end = start + batch_size if start + batch_size < len(input_features) else len(input_features)
        xx = torch.tensor(input_features[start:end],dtype = torch.float,requires_grad = True)
        yy = torch.tensor(labels[start:end],dtype = torch.float,requires_grad = True)
        prediction = my_nn(xx)
        loss = cost(prediction,yy)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        batch_loss.append(loss.data.numpy())

    if i % 100 == 0:
        losses.append(np.mean(batch_loss))
        print(i,np.mean(batch_loss))

#--------------------------------------5、预测天气并可视化---------------------------------
#预测训练结果
x = torch.tensor(input_features,dtype = torch.float)
predict = my_nn(x).data.numpy()

months = features[:,feature_list.index('month')]
days = features[:,feature_list.index('day')]
years = features[:,feature_list.index('year')]
test_dates = [str(int(year))+'-'+str(int(month))+'-'+str(int(day)) for year,month,day in zip(years,months,days)]
test_dates = [datetime.datetime.strptime(date,'%Y-%m-%d') for date in test_dates]

#预测天气温度表
predictions_data = pd.DataFrame(data = {'date':test_dates,'prediction':predict.reshape(-1)})
#实际天气温度表
true_data = pd.DataFrame(data={'date':dates,'actual':labels})

#绘制‘预测’和‘实际’的温度变化折线图
plt.plot(predictions_data['date'],predictions_data['prediction'],'ro',label = 'prediction')
plt.plot(true_data['date'],true_data['actual'],'b-',label = 'actual')


plt.xticks(rotation = '60')
plt.legend()
plt.xlabel('Date');plt.ylabel('Maximum Temparature(F)' );plt.title('Actual and Predicted Values');
plt.show()