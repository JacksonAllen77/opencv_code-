import cv2
import numpy as np
import matplotlib.pyplot as plt
# 以灰度图加载两张图片
img1 = cv2.imread('F:/Pycharm/opencv/opencv/OpenCV-main/01_Picture/19_Box.png',0)
img2 = cv2.imread('F:/Pycharm/opencv/opencv/OpenCV-main/01_Picture/20_Box_in_scene.png',0)
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
cv_show('19_Box.png',img1)
cv_show('20_Box_in_scene.png',img2)

sift = cv2.xfeatures2d.SIFT_create() # 创建SIFT特征检测器
kp1, des1 = sift.detectAndCompute(img1,None) # 计算两张图的特征点、特征向量
kp2, des2 = sift.detectAndCompute(img2,None)

# K对最佳匹配
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)  # k 参数可选，可以一个点跟它最近的k个点可选

good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:  # m.distance 与 n.distance 比值小于 0.75，这是自己设定的过滤条件
        good.append([m])

img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
cv_show('img3', img3)