import cv2
import matplotlib.pyplot as plt
import numpy as np

def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  # 使用 destroyAllWindows() 关闭所有窗口


image_path = '/home/ro/ML-DL-CV/opencv_code/OpenCV-main/01_Picture/01_cat.jpg'  # 给定图像路径
img = cv2.imread(image_path) # 读取图像
print(f"图像的数据类型{type(img)}、图像的像素点个数{img.size}、图像的形状: {img.shape}、图像的元素类型: {img.dtype}")
cv_show("helloworld", img) # 显示图像