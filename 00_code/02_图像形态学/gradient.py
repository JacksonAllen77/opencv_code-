
import cv2 #opencv的缩写为cv2
import matplotlib.pyplot as plt # matplotlib库用于绘图展示
import numpy as np   # numpy数值计算工具包

#梯度运算——膨胀减去腐蚀【得出轮廓】
pie = cv2.imread('/home/ro/ML-DL-CV/opencv_code/OpenCV-main/01_Picture/06_pie.png')
kernel = np.ones((7,7),np.uint8) # 创建卷积核
dilate = cv2.dilate(pie,kernel,iterations=5) # 执行5次膨胀
erosion = cv2.erode(pie,kernel,iterations=5) # 执行5次腐蚀
res = np.hstack((dilate,erosion)) # 将两个图像进行水平拼接
cv2.imshow('res',res)
cv2.waitKey(0)
cv2.destroyAllWindows()

gradient = cv2.morphologyEx(pie,cv2.MORPH_GRADIENT,kernel)#对图像以卷积核执行梯度运算
cv2.imshow('gradient',gradient)
cv2.waitKey(0)
cv2.destroyAllWindows()