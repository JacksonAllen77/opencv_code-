import cv2 #opencv的缩写为cv2
import matplotlib.pyplot as plt # matplotlib库用于绘图展示
import numpy as np   # numpy数值计算工具包
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey()
    cv2.destroyAllWindows()

img = cv2.imread('/home/ro/ML-DL-CV/opencv_code/OpenCV-main/01_Picture/04_LenaNoise.png')
cv_show("img",img)#显示原始图像

blur = cv2.blur(img,(3,3))
aussian = cv2.GaussianBlur(img,(3,3),1)
median = cv2.medianBlur(img,3)

res = np.hstack((blur,aussian,median)) # 矩阵横着拼接
#res = np.vstack((blur,aussian,median)) # 矩阵竖着拼接
print(res)
cv_show("res",res)