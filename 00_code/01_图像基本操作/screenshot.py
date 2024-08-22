import cv2
import matplotlib.pyplot as plt
import numpy as np

def cv_show(name, img):
    # 显示图像
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  # 使用 destroyAllWindows() 关闭所有窗口
image_path = '/home/ro/ML-DL-CV/opencv_code/OpenCV-main/01_Picture/01_cat.jpg'  # 给定图像路径
img=cv2.imread(image_path)
screen_cat= img[100:200, 200:450]
cv_show("screen",screen_cat)