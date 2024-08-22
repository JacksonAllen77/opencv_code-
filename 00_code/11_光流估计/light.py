import numpy as np
import cv2

cap = cv2.VideoCapture('F:/Pycharm/opencv/opencv/opencv_code/OpenCV-main/02_Video/01_Foreground.avi')

# 角点检测所需参数
# 如果不限制角点最大数量，速度就会有些慢，达不到实时的效果
# 品质因子会筛选角点，品质因子设置的越大，得到的角点越少
# 设置最小距离，判断该距离效果最好的角点；对其余角点进行过滤
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7)

# lucas-kanada参数
# winSize：窗口大小 maxLevel：金字塔层数
lk_params = dict(winSize=(15, 15),
                 maxLevel=2)

# 随即颜色条
color = np.random.randint(0, 255, (100, 3))

# 拿到第一帧图像，对其进行灰度图转换
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY) # 第一帧图像读取

# cv2.goodFeaturesToTrack函数返回所有检测特征点，需要输入：图像，角点最大数量(效率)，品质因子(特征值越大的越好来筛选)
# 距离相当于这区间有比这个角点强的，就不要这个弱的了
# **变量 作为传入参数，是用来传入不定长的变量
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)  # 拿到第一帧的角点，后面视频中是对第一帧的角点进行追踪

# 创建一个 mask
mask = np.zeros_like(old_frame)

while (True): # 采用while循环读取视频中的每一帧，以灰度图读取
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #当前帧图像读取

    # 需要传入前一帧和当前图像以及前一帧检测到的角点
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # st=1 表示
    good_new = p1[st == 1]  # st==1 表示找到的特征点，没找到的特征点就不要了
    good_old = p0[st == 1]

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
    img = cv2.add(frame, mask)

    cv2.imshow('frame', img)
    k = cv2.waitKey(150) & 0xff
    if k == 27:
        break

    # 更新——将当前帧作为下一帧的过去
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv2.destroyAllWindows()
cap.release()