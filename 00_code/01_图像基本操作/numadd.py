import cv2 #opencv的缩写为cv2
import matplotlib.pyplot as plt # matplotlib库用于绘图展示
import numpy as np   # numpy数值计算工具包

image_path = '/home/ro/ML-DL-CV/opencv_code/OpenCV-main/01_Picture/01_cat.jpg'  # 给定图像路径
img_cat = cv2.imread(image_path) # 读取图像

img_cat2=img_cat+10 # 将img图像的每一个像素点均+10
print(img_cat[:5,:,0])# 打印 img_cat 图像前 5 行的红色通道值
print(img_cat2[:5,:,0])# 打印 img_cat2 图像前 5 行的红色通道值

print("numpy数值计算")
print((img_cat+img_cat2)[:5,:,0])  # 0-255 若相加越界后 294 用 294%256 取余数 38
print("cv数值计算")
print(cv2.add(img_cat,img_cat2)[:5,:,0]) # cv2.add 是越界后取最大值 255