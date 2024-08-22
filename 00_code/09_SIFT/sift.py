import cv2
import numpy as np

img = cv2.imread('F:/Pycharm/opencv/opencv/OpenCV-main/01_Picture/18_House.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 得到特侦点
sift = cv2.xfeatures2d.SIFT_create()  # 将 SIFT 算法实例化出来
kp = sift.detect(gray, None) # 把灰度图传进去，得到特征点、关键点
img = cv2.drawKeypoints(gray, kp, img) # 绘制关键点
cv2.imshow('drawKeypoints', img) #显示图像
cv2.waitKey(0)
cv2.destroyAllWindows()
# 计算特征
kp, des = sift.compute(gray, kp) # 计算出图像的关键点&描述矩阵
print(np.array(kp).shape) # 输出关键点数量
print(des.shape) # 输出关键点维度信息
print(des[0])    # 输出第一个关键点的描述信息