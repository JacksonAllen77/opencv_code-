import cv2  # opencv的缩写为cv2
import matplotlib.pyplot as plt  # matplotlib库用于绘图展示
import numpy as np  # numpy数值计算工具包
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread('/home/ro/ML-DL-CV/opencv_code/OpenCV-main/01_Picture/04_LenaNoise.png')
cv_show("img", img)  # 显示原始图像

# 中值滤波
# 排序后拿中值替代中间元素值的大小
median = cv2.medianBlur(img,3) # 3 表示卷积核的大小为(3,3)
cv_show("median", median)  # 显示原始图像