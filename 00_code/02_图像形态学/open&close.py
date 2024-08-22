import cv2 #opencv的缩写为cv2
import matplotlib.pyplot as plt # matplotlib库用于绘图展示
import numpy as np   # numpy数值计算工具包


# # 开运算——先腐蚀，再膨胀
img = cv2.imread('/home/ro/ML-DL-CV/opencv_code/OpenCV-main/01_Picture/05_Dige.png')
# kernel = np.ones((5,5),np.uint8)
# opening = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel) #对图像以卷积核执行开运算
# cv2.imshow('opening',opening)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 闭运算——先膨胀，再腐蚀
kernel = np.ones((5,5),np.uint8)
closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
cv2.imshow('closing',closing)
cv2.waitKey(0)
cv2.destroyAllWindows()