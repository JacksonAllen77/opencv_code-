import cv2 #opencv的缩写为cv2
import matplotlib.pyplot as plt # matplotlib库用于绘图展示
import numpy as np   # numpy数值计算工具包

hsv = cv2.imread('/home/ro/ML-DL-CV/opencv_code/OpenCV-main/01_Picture/01_cat.jpg', cv2.COLOR_BGR2HSV)
cv2.imshow('hsv',hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()