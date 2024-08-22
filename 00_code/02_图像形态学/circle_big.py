import cv2 #opencv的缩写为cv2
import matplotlib.pyplot as plt # matplotlib库用于绘图展示
import numpy as np   # numpy数值计算工具包

pie = cv2.imread('/home/ro/ML-DL-CV/opencv_code/OpenCV-main/01_Picture/06_pie.png')
cv2.imshow('pie',pie)
cv2.waitKey(0)
cv2.destroyAllWindows()

kernel = np.ones((30,30),np.uint8)
dilate_1 = cv2.dilate(pie,kernel,iterations=1)
dilate_2 = cv2.dilate(pie,kernel,iterations=2)
dilate_3 = cv2.dilate(pie,kernel,iterations=3)
res = np.hstack((dilate_1,dilate_2,dilate_3))
cv2.imshow('res',res)
cv2.waitKey(0)
cv2.destroyAllWindows()