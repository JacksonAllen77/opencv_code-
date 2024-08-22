import  cv2
import numpy as np
import matplotlib.pyplot as plt
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  # 使用 destroyAllWindows() 关闭所有窗口

# 创建掩码
img=cv2.imread("F:/Pycharm/opencv/opencv/OpenCV-main/01_Picture/01_cat.jpg")
mask=np.zeros(img.shape[:2],np.uint8)
mask[100:300,100:400]=255
cv_show("mask",mask)
img=cv2.imread("F:/Pycharm/opencv/opencv/OpenCV-main/01_Picture/01_cat.jpg",0)
cv_show("img",img)
# 将猫的照片与掩码进行与操作
masked_img=cv2.bitwise_and(img,img,mask=mask)
cv_show("masked_img",masked_img)
# 统计直方图
hist_full=cv2.calcHist([img],[0],None,[256],[0,256])#统计原图的直方图
hist_mask=cv2.calcHist([img],[0],mask,[256],[0,256])#统计掩码状态下的直方图
#matplotlib出图
plt.subplot(221),plt.imshow(img,"gray")
plt.subplot(222),plt.imshow(mask,"gray")
plt.subplot(223),plt.imshow(masked_img,"gray")
plt.subplot(224),plt.plot(hist_full),plt.plot(hist_mask)
plt.xlim([0,256])
plt.show()
