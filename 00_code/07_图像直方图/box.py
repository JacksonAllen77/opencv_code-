import  cv2
import numpy as np
import matplotlib.pyplot as plt
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  # 使用 destroyAllWindows() 关闭所有窗口

img=cv2.imread("F:/Pycharm/opencv/opencv/OpenCV-main/01_Picture/01_cat.jpg") # 表示以灰度图进行读取
color=("b","g","r")
for i,col in enumerate(color):
    histr=cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color=col)
    plt.xlim([0,256])
    plt.show()