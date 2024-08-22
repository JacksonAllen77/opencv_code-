import cv2 #opencv的缩写为cv2
import matplotlib.pyplot as plt # matplotlib库用于绘图展示
import numpy as np   # numpy数值计算工具包
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey()
    cv2.destroyAllWindows()

img = cv2.imread('/home/ro/ML-DL-CV/opencv_code/OpenCV-main/01_Picture/10_contours.png')
cv_show('img',img)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # 灰度处理
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY) # 二值化处理
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # 利用边缘检测函数定位轮廓
#轮廓特诊提取
cnt = contours[0] # 通过轮廓索引，拿到该索引对应的轮廓特征
print(cv2.contourArea(cnt)) # 该轮廓的面积
print(cv2.arcLength(cnt,True)) # 该轮廓的周长，True表示闭合的
