import cv2 #opencv的缩写为cv2
import matplotlib.pyplot as plt # matplotlib库用于绘图展示
import numpy as np   # numpy数值计算工具包
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey()
    cv2.destroyAllWindows()

img = cv2.imread('/home/ro/ML-DL-CV/opencv_code/OpenCV-main/01_Picture/11_contours2.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # 灰度处理
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY) # 二值化处理
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # 利用边缘检测函数定位轮廓

# 进行原图备份，对其中一个轮廓进行绘制
cnt = contours[0] # 取出其中一个轮廓
draw_img = img.copy() # 对原图进行备份
res = cv2.drawContours(draw_img,[cnt],-1,(0,0,255),2) # 绘制取出的轮廓线
cv_show('res',res)

# 以周长的百分比相关作为阈值，
epsilon = 0.1 * cv2.arcLength(cnt,True) # 周长的百分比，这里用 0.1 的周长作阈值
approx = cv2.approxPolyDP(cnt,epsilon,True) # 以阈值为参考值，进行近似处理
draw_img = img.copy() # 对原图进行备份
res = cv2.drawContours(draw_img,[approx],-1,(0,0,255),2)
cv_show('res',res)