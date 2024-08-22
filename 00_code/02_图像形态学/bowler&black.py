import cv2 #opencv的缩写为cv2
import matplotlib.pyplot as plt # matplotlib库用于绘图展示
import numpy as np   # numpy数值计算工具包

#礼帽运算——原始减去开运算【得出毛刺】
img = cv2.imread('/home/ro/ML-DL-CV/opencv_code/OpenCV-main/01_Picture/05_Dige.png')
kernel = np.ones((5,5),np.uint8)
blackhat = cv2.morphologyEx(img,cv2.MORPH_BLACKHAT,kernel)
cv2.imshow('blackhat',blackhat)
cv2.waitKey(0)
cv2.destroyAllWindows()

