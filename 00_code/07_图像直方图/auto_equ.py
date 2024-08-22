import  cv2
import numpy as np
import matplotlib.pyplot as plt
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  # 使用 destroyAllWindows() 关闭所有窗口

img=cv2.imread("F:/Pycharm/opencv/opencv/OpenCV-main/01_Picture/16_Clahe.jpg",0) # 表示以灰度图进行读取

# 对图像执行分块自适应均衡化
clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))#对原图进行自适应分块均衡化
equ=cv2.equalizeHist(img)
res_clahe=clahe.apply(img)
res=np.hstack((img,equ,res_clahe))#原图与全局与自适应对比
cv_show("res",res)