import numpy as np
import cv2

# 经典的测试视频
cap = cv2.VideoCapture('F:/Pycharm/opencv/opencv/OpenCV-main/02_Video/01_Foreground.avi')

# 初始化前一帧
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)  # 转为灰度图

while True:
    ret, frame = cap.read()  # 读取当前帧
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转为灰度图

    # 计算当前帧与前一帧的差异
    frame_diff = cv2.absdiff(prev_gray, gray)

    # 对差异图进行阈值处理以得到二值图像
    _, fgmask = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

    # 形态学开运算去噪点
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    # 寻找轮廓
    _,contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 遍历每一个轮廓，对轮廓进行判断
    for c in contours:
        # 计算轮廓的周长
        perimeter = cv2.arcLength(c, True)

        if perimeter > 188:
            # 找到一个直矩形 (不会旋转)
            x, y, w, h = cv2.boundingRect(c)
            # 画出这个矩形
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    cv2.imshow('fgmask', fgmask)

    # 按键退出
    k = cv2.waitKey(150) & 0xff
    if k == 27:
        break

    # 更新前一帧
    prev_gray = gray

cap.release()
cv2.destroyAllWindows()
