import cv2
import matplotlib.pyplot as plt
import numpy as np

def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  # 使用 destroyAllWindows() 关闭所有窗口


image_path = '/home/ro/ML-DL-CV/opencv_code/OpenCV-main/01_Picture/01_cat.jpg'  # 给定图像路径
img = cv2.imread(image_path) # 读取图像
b,g,r=cv2.split(img)
# 只保留B通道
cur_img=img.copy()
cur_img[:,:,1]=0
cur_img[:,:,2]=0
cv_show("B_cat",cur_img)

# 只保留G通道
cur_img1=img.copy()
cur_img1[:,:,0]=0
cur_img1[:,:,2]=0
cv_show("G_cat",cur_img1)

# 只保留R通道
cur_img2=img.copy()
cur_img2[:,:,0]=0
cur_img2[:,:,1]=0
cv_show("R_cat",cur_img2)

