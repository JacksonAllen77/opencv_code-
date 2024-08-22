import cv2 #opencv的缩写为cv2
import matplotlib.pyplot as plt # matplotlib库用于绘图展示
import numpy as np   # numpy数值计算工具包

template=cv2.imread("/home/ro/ML-DL-CV/opencv_code/OpenCV-main/01_Picture/12_Face.jpg")
img=cv2.imread("/home/ro/ML-DL-CV/opencv_code/OpenCV-main/01_Picture/13_Lena.jpg")
h, w = template.shape[:2] # 获得模板的宽和高
# template.shape 返回一个包含图像维度的元组。对于一个彩色图像，shape 元组通常是 (高度, 宽度, 通道数)
print(img.shape)
print(template.shape)
# 采用模板匹配的方法，对该模板进行检测【实际使用时，使用归一化的方法结果更可靠】
methods = ['cv2.TM_CCOEFF','cv2.TM_CCOEFF_NORMED','cv2.TM_CCORR',
          'cv2.TM_CCORR_NORMED','cv2.TM_SQDIFF','cv2.TM_SQDIFF_NORMED']
res = cv2.matchTemplate(img, template, cv2.TM_SQDIFF) #采用cv2.TM_SQDIFF方法
print(res.shape) # 返回的矩阵大小 (A-a+1)x(B-b+1)
min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(res) # 返回模板匹配后最小值、最大值的位置
print(min_val) # cv2.TM_SQDIFF方法中，越小的值表示像素点的差异越小
print(max_val)
print(min_loc) # 当获得最小值对应的模板左上角的位置，加上模板自身的长、宽，可以在原图像中画出最匹配的区域
print(max_loc)