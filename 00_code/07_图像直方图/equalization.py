import  cv2
import numpy as np
import matplotlib.pyplot as plt
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  # 使用 destroyAllWindows() 关闭所有窗口

img=cv2.imread("F:/Pycharm/opencv/opencv/OpenCV-main/01_Picture/01_cat.jpg",0) # 表示以灰度图进行读取
plt.hist(img.ravel(),256)
plt.show()
# 对图像执行直方图均衡化
equ=cv2.equalizeHist(img)
plt.hist(equ.ravel(),256) # 直方图绘制
plt.show()
res=np.hstack((img,equ)) # 原图与均衡化图比较
cv_show("res",res)