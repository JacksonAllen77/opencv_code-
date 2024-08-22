import cv2 #opencv的缩写为cv2
import matplotlib.pyplot as plt # matplotlib库用于绘图展示
import numpy as np   # numpy数值计算工具包

image_path = '/home/ro/ML-DL-CV/opencv_code/OpenCV-main/01_Picture/01_cat.jpg'  # 给定图像路径
img = cv2.imread(image_path) # 读取图像
# 对图像进行填充
top_size,bottom_size,left_size,right_size = (50,50,50,50)  # 上下左右均填充50个像素块
# 不同填充方式的填充结果
# 方式一：复制法
replicate = cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,borderType=cv2.BORDER_REPLICATE)
# 方式二：反射法
reflect = cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,cv2.BORDER_REFLECT)
# 方式三：反射法二(不要最边缘的像素)
reflect101 = cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,cv2.BORDER_REFLECT_101)
# 方式四：外包装法
wrap = cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,borderType=cv2.BORDER_WRAP)
# 方式五：常量法
constant = cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,cv2.BORDER_CONSTANT,value=0)

import matplotlib.pyplot as plt
plt.subplot(231), plt.imshow(img,'gray'), plt.title('ORIGINAL')
plt.subplot(232), plt.imshow(replicate,'gray'), plt.title('REPLICATE')
plt.subplot(233), plt.imshow(reflect,'gray'), plt.title('REPLECT')
plt.subplot(234), plt.imshow(wrap,'gray'),plt.title('REFLECT_101')
plt.subplot(235), plt.imshow(wrap,'gray'),plt.title('WRAP')
plt.subplot(236), plt.imshow(constant,'gray'),plt.title('CONSTAVI')

plt.show()