import cv2 #opencv的缩写为cv2
import matplotlib.pyplot as plt # matplotlib库用于绘图展示
import numpy as np   # numpy数值计算工具包

def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey()
    cv2.destroyAllWindows()

img = cv2.imread('/home/ro/ML-DL-CV/opencv_code/OpenCV-main/01_Picture/09_AM.png')
cv_show('img',img)
print(img.shape)
# 执行一次高斯上采样【增加分辨率】
up = cv2.pyrUp(img)
cv_show('up',up)
print(up.shape)
# 执行一次高斯下采样【减小分辨率】
down = cv2.pyrDown(img)
cv_show('down',down)
print(down.shape)

# 将原图先执行一次上采样，再执行一次下采样，然后与原图比较
up = cv2.pyrUp(img)
up_down = cv2.pyrDown(up) # 先上采样再下采样
cv_show('img,up_down',np.hstack((img,up_down)))# 将原图&处理后图像进行拼接，以进行比较
# 原图&处理后图像差异计算
cv_show("img-up_down",img-up_down)# 差异计算
#亮区域：在差异图像中，亮度较高的区域表示两个图像之间的差异较大。表示信息失真部分
#暗区域：在差异图像中，亮度较低的区域表示两个图像之间的差异较小。表示信息存留部分