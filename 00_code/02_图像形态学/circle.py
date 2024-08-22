import cv2 #opencv的缩写为cv2
import matplotlib.pyplot as plt # matplotlib库用于绘图展示
import numpy as np   # numpy数值计算工具包

pie = cv2.imread('/home/ro/ML-DL-CV/opencv_code/OpenCV-main/01_Picture/06_pie.png')
cv2.imshow('pie',pie)
cv2.waitKey(0)
cv2.destroyAllWindows()

kernel = np.ones((30,30),np.uint8)
erosion_1 = cv2.erode(pie,kernel,iterations=1)
erosion_2 = cv2.erode(pie,kernel,iterations=2)
erosion_3 = cv2.erode(pie,kernel,iterations=3)
res = np.hstack((erosion_1,erosion_2,erosion_3))#将3个腐蚀后的图像水平拼接【np.hstack用于将多个图像沿水平方向合并】
cv2.imshow('res',res)
cv2.waitKey(0)
cv2.destroyAllWindows()