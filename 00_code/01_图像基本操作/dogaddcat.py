import cv2 #opencv的缩写为cv2
import matplotlib.pyplot as plt # matplotlib库用于绘图展示
import numpy as np   # numpy数值计算工具包

def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  # 使用 destroyAllWindows() 关闭所有窗口

cat_path = '/home/ro/ML-DL-CV/opencv_code/OpenCV-main/01_Picture/01_cat.jpg'  # 给定图像路径
img_cat = cv2.imread(cat_path) # 读取图像
print(img_cat.shape)
dog_path = '/home/ro/ML-DL-CV/opencv_code/OpenCV-main/01_Picture/03_dog.jpg'  # 给定图像路径
img_dog = cv2.imread(dog_path) # 读取图像
print(img_dog.shape)
# 图像变换
# 狗和猫的尺寸不同——分别为(414, 500, 3)、(429, 499, 3)
img_dog = cv2.resize(img_dog,(500,414))#将狗的图片尺寸重写设置为宽度500，高度414
print(img_dog.shape)

# 图像融合
res = cv2.addWeighted(img_cat,0.4,img_dog,0.6,0) # img_cat的权重为0.4，img_dog的权重为 0.6,亮度集提亮值为0
cv_show("res",res)