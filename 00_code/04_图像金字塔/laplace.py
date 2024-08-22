import cv2 #opencv的缩写为cv2
import matplotlib.pyplot as plt # matplotlib库用于绘图展示
import numpy as np   # numpy数值计算工具包

def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey()
    cv2.destroyAllWindows()

img = cv2.imread('/home/ro/ML-DL-CV/opencv_code/OpenCV-main/01_Picture/09_AM.png')
cv_show('img',img)

down_up1=cv2.pyrDown(cv2.pyrUp(img))
laplace1=img-down_up1
down_up2=cv2.pyrDown(cv2.pyrUp(down_up1))
laplace2=down_up2-down_up1
down_up3=cv2.pyrDown(cv2.pyrUp(down_up2))
laplace3=down_up3-down_up2
cv_show('img,laplace1,laplace2',np.hstack((img,laplace1,laplace2)))# 将原图&处理后图像进行拼接，以进行比较