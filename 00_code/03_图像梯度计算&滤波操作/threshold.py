import cv2 #opencv的缩写为cv2
import matplotlib.pyplot as plt # matplotlib库用于绘图展示
import numpy as np   # numpy数值计算工具包
img = cv2.imread('/home/ro/ML-DL-CV/opencv_code/OpenCV-main/01_Picture/01_cat.jpg',cv2.IMREAD_COLOR)
img_gray = cv2.imread('/home/ro/ML-DL-CV/opencv_code/OpenCV-main/01_Picture/01_cat.jpg',cv2.IMREAD_GRAYSCALE)
ret, thresh1 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)#阈值为127，像素值大于阈值设为最大值（255），否则设为0【二值化处理】
print(ret)
ret, thresh2 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV) #像素值大于阈值设为0，否则设为255【反二值化处理】
print(ret)
ret, thresh3 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TRUNC)#像素值大于阈值的设为阈值，其余不变【阈值截断】
print(ret)
ret, thresh4 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO)#像素值大于阈值保留，其余设为0。【阈值0处理】
print(ret)
ret, thresh5 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO_INV)#像素值大于阈值设为0，其余保留【反向阈值为0处理】
print(ret)

titles = ['original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV'] # 图片标题列表
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5] #图像列表

for i in range(6):#生成子图网格
    plt.subplot(2,3,i+1), plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()