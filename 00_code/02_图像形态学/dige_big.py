import cv2 #opencv的缩写为cv2
import matplotlib.pyplot as plt # matplotlib库用于绘图展示
import numpy as np   # numpy数值计算工具包

def cv_show(name, img):
    # 显示图像
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  # 使用 destroyAllWindows() 关闭所有窗口

img = cv2.imread('/home/ro/ML-DL-CV/opencv_code/OpenCV-main/01_Picture/05_Dige.png')
cv_show("origin",img)

# 先腐蚀 后膨胀，抵消腐蚀造成的损害
kernel = np.ones((3,3),np.uint8)
dige_erosion = cv2.erode(img,kernel,iterations=1)
cv2.imshow('erosion',dige_erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()

kernel = np.ones((3,3),np.uint8)
dige_dilate = cv2.dilate(dige_erosion,kernel,iterations=1)
cv2.imshow('dilate',dige_dilate)
cv2.waitKey(0)
cv2.destroyAllWindows()