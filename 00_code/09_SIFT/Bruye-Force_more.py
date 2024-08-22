import cv2
import numpy as np

# 以灰度图加载两张图片
img1 = cv2.imread('F:/Pycharm/opencv/opencv/OpenCV-main/01_Picture/19_Box.png', 0)
img2 = cv2.imread('F:/Pycharm/opencv/opencv/OpenCV-main/01_Picture/20_Box_in_scene.png', 0)

def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

cv_show('19_Box.png', img1)
cv_show('20_Box_in_scene.png', img2)

# 创建SIFT特征检测器
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# 使用交叉验证的蛮力匹配器
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# 通过Lowe's ratio test筛选匹配点
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

# 确保有足够的好匹配点
if len(good) > 4:
    # 获取匹配点的坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2) #获取第一个图像的特征点坐标
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2) #获取第二个图像的特征点坐标

    # 使用RANSAC算法计算单应性矩阵并过滤离群点
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0) # 计算图像之间的单应性矩阵（M）。这是通过匹配点对来推测从 img1 到 img2 的几何变换。
    matchesMask = mask.ravel().tolist()

    # 使用过滤后的匹配点绘制匹配结果
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, [good[i] for i in range(len(good)) if matchesMask[i]], None, flags=2) # 绘制匹配结果
    cv_show('Filtered Matches', img3)
else:
    print("Not enough matches are found - {}/{}".format(len(good), 4)) # 如果匹配点不足4个,输出错误信息
