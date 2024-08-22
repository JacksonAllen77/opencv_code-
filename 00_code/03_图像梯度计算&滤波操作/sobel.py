import cv2 #opencv的缩写为cv2
import matplotlib.pyplot as plt # matplotlib库用于绘图展示
import numpy as np   # numpy数值计算工具包
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey()
    cv2.destroyAllWindows()

pie = cv2.imread('/home/ro/ML-DL-CV/opencv_code/OpenCV-main/01_Picture/06_pie.png')
cv_show("pie",pie)#显示原始图像

#opencv默认是截断操作，即小于0的按0算，大于255的按255算
#而右侧边缘为左边白，右边黑。根据sober算子【右减左】，则为负数，截断为0；因此只含有左半边轮廓
sobelx = cv2.Sobel(pie,cv2.CV_64F,1,0,ksize=3) # 1,0 表示只算水平方向梯度,3表示卷积核大小[cv2.CV_64F：图像的深度，指定输出图像的像素数据类型。使用 64 位浮点型可以保证计算精度。]
cv_show('sobelx',sobelx)
#优化算法——当为附属时，取绝对值
sobelx = cv2.Sobel(pie,cv2.CV_64F,1,0,ksize=3)
sobelx = cv2.convertScaleAbs(sobelx) # 优化算法——当为负数时，取绝对值
cv_show('sobelx',sobelx)

sobely = cv2.Sobel(pie,cv2.CV_64F,0,1,ksize=3) # 1,0 只算 y 方向梯度
sobely = cv2.convertScaleAbs(sobely) # 取负数时，取绝对值
cv_show('sobely',sobely)

# 计算 x 和 y 后，再求和
sobelxy = cv2.addWeighted(sobelx,0.5,sobely,0.5,0) # 0是偏置项
cv_show('sobelxy',sobelxy)