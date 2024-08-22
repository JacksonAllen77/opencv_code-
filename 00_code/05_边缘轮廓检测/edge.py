import cv2 #opencv的缩写为cv2
import matplotlib.pyplot as plt # matplotlib库用于绘图展示
import numpy as np   # numpy数值计算工具包
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey()
    cv2.destroyAllWindows()

img = cv2.imread('/home/ro/ML-DL-CV/opencv_code/OpenCV-main/01_Picture/08_Car.png')
cv_show('img',img)
# 为了更好的实现边缘检测，彩色图片先进行灰度处理、再进行二值化处理
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # 灰度处理
cv_show('gray',gray)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY) # 二值化处理


# 做完二值后，再用图像轮廓检测函数再去做
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# 注意需要copy，要不原图会变。。。
draw_img = img.copy() # 若不用拷贝后的，而是用原图画轮廓，则画轮廓图绘把原始的输入图像重写，覆盖掉
# 传入参数：图像、轮廓、轮廓索引(自适应，画所有轮廓)，颜色模式，线条厚度
res = cv2.drawContours(draw_img,contours,-1,(0,0,255),2)#用红色描绘轮廓
cv_show('res',res)