import cv2 #opencv的缩写为cv2
import matplotlib.pyplot as plt # matplotlib库用于绘图展示
import numpy as np   # numpy数值计算工具包
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey()
    cv2.destroyAllWindows()

img = cv2.imread('/home/ro/ML-DL-CV/opencv_code/OpenCV-main/01_Picture/07_Lena.jpg', cv2.IMREAD_GRAYSCALE)
cv_show("img",img)#显示原始图像
v1 = cv2.Canny(img,80,150) # 第二个参数为minVal，第三个参数为maxVal
v2 = cv2.Canny(img,50,100)
res = np.hstack((v1,v2))
cv_show('res',res)

img2 = cv2.imread('/home/ro/ML-DL-CV/opencv_code/OpenCV-main/01_Picture/08_Car.png', cv2.IMREAD_GRAYSCALE)
cv_show("img2",img2)#显示原始图像

# 不同严格度状态下的边界
v3 = cv2.Canny(img2,120,250) # 第二个参数为minVal，第三个参数为maxVal
v4 = cv2.Canny(img2,50,100)

res2 = np.hstack((v3,v4))
cv_show('res2',res2)