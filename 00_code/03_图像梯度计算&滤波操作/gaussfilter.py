import cv2  # opencv的缩写为cv2
import matplotlib.pyplot as plt  # matplotlib库用于绘图展示
import numpy as np  # numpy数值计算工具包
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread('/home/ro/ML-DL-CV/opencv_code/OpenCV-main/01_Picture/04_LenaNoise.png')
cv_show("img", img)  # 显示原始图像

# 高斯函数，越接近均值时，它的概率越大。
# 离中心值越近的，它的权重越大，离中心值越远的，它的权重越小。
aussian = cv2.GaussianBlur(img,(3,3),1) # (3,3) 为卷积核的大小,1为高斯函数的标准差
#标准差控制高斯函数的宽度，进而影响滤波的强度。标准差越大，滤波器的影响范围越广，图像会越平滑。标准差越小，滤波器的影响范围越小，平滑效果较轻微。
cv_show("aussian", aussian)  # 显示原始图像