import cv2 #opencv的缩写为cv2
import matplotlib.pyplot as plt # matplotlib库用于绘图展示
import numpy as np   # numpy数值计算工具包

img_rgb = cv2.imread('/home/ro/ML-DL-CV/opencv_code/OpenCV-main/01_Picture/14_Mario.jpg')
img_gray = cv2.cvtColor(img_rgb,cv2.COLOR_BGR2GRAY)
print('img_gray.shape：',img_gray.shape)
template = cv2.imread('/home/ro/ML-DL-CV/opencv_code/OpenCV-main/01_Picture/15_Mario_coin.jpg',0)
print('template.shape：',template.shape)
h, w = template.shape[:2]

res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED) # res 是返回每一个小块窗口得到的结果值
print('res.shape：',res.shape)

# 设定一个阈值，若匹配度高于这个阈值则被标记
threshold = 0.8
# 取匹配程度大于 80% 的坐标
loc = np.where(res >= threshold) # np.where 使得返回 res 矩阵中值大于 0.8 的索引，即坐标
print('type(loc):',type(loc)) # loc 为元组类型
print('len(loc):',len(loc))  # loc 元组有两个值
print('len(loc[0]):',len(loc[0]),'len(loc[1]):',len(loc[1]))   # loc 元组每个值 120 个元素
print('type(loc[0]):',type(loc[0]),'type(loc[1]):',type(loc[1])) # loc 元组每个值的类型为 numpy.array
print("loc[::-1]：",loc[::-1]) # loc[::-1] 表示顺序取反，即第二个 numpy.array 放在第一个 numpy.array 前面

i = 0
# zip函数为打包为元组的列表，例 a = [1,2,3] b = [4,5,6] zip(a,b) 为 [(1, 4), (2, 5), (3, 6)]
for pt in zip(*loc[::-1]): # 当用 *b 作为传入参数时, b 可以为列表、元组、集合，zip使得元组中两个 numpy.array 进行配对
    bottom_right = (pt[0] + w, pt[1] + h)
    cv2.rectangle(img_rgb, pt, bottom_right, (0,0,255),2)
    i = i + 1
print('i:',i)

cv2.imshow('img_rgb',img_rgb)
cv2.waitKey(0)