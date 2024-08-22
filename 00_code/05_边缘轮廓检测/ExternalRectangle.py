import cv2 #opencv的缩写为cv2
import matplotlib.pyplot as plt # matplotlib库用于绘图展示
import numpy as np   # numpy数值计算工具包
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey()
    cv2.destroyAllWindows()

img = cv2.imread('F:/Pycharm/opencv/opencv/OpenCV-main/01_Picture/10_contours.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY) # 大于17的取255，小于127的取0
binary, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cnt = contours[2]
draw_img = img.copy() # 对原图进行备份

# 外接矩形
x,y,w,h = cv2.boundingRect(cnt) # 计算轮廓的外接矩形，可以得到矩形四个坐标点的相关信息
img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255),2) #绘制矩形，(x, y)为矩形的左上角坐标；(x + w, y + h)为矩形的右下角坐标
cv_show('img',img)
# 外接圆
(x,y),redius = cv2.minEnclosingCircle(cnt) # 计算轮廓的外接圆，可以得到圆的圆心和半径的相关信息
center = (int(x),int(y))
redius = int(redius)
img = cv2.circle(draw_img,center,redius,(0,255,0),2)#绘制圆
cv_show('img',img)

#可以将轮廓和边界图形进行计算操作
area = cv2.contourArea(cnt) #对轮廓计算面积
rect_area = w * h #计算矩形的面积
extent = float(area) / rect_area
print('轮廓面具与边界矩形比：',extent)