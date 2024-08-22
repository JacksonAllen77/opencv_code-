import cv2 #opencv的缩写为cv2
import matplotlib.pyplot as plt # matplotlib库用于绘图展示
import numpy as np   # numpy数值计算工具包
img = cv2.imread('F:/Pycharm/opencv/opencv/OpenCV-main/01_Picture/04_LenaNoise.png')
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
blur = cv2.blur(img,(3,3)) # (3,3) 为核的大小，通常情况核都是奇数 3、5、7
cv2.imshow('blur',blur)
cv2.waitKey(0)
cv2.destroyAllWindows()