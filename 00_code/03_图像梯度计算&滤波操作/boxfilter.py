import cv2  # opencv的缩写为cv2
import matplotlib.pyplot as plt  # matplotlib库用于绘图展示
import numpy as np  # numpy数值计算工具包


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread('/home/ro/ML-DL-CV/opencv_code/OpenCV-main/01_Picture/04_LenaNoise.png')
cv_show("img", img)  # 显示原始图像

# 方框滤波
# 基本和均值一样，可以选择归一化
# 在 Python 中 -1 表示自适应填充对应的值，这里的 -1 表示与颜色通道数自适应一样
box = cv2.boxFilter(img,-1,(3,3),normalize=True)  # 归一化，得到的结果和均值滤波一模一样
cv_show("box", box)  # 显示原始图像
box = cv2.boxFilter(img,-1,(3,3),normalize=False)  # 未作归一化，越界的值取 255
cv_show("box", box)  # 显示原始图像