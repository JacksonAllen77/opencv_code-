import cv2 #opencv的缩写为cv2
import matplotlib.pyplot as plt # matplotlib库用于绘图展示
import numpy as np   #numpy数值计算工具包



video_path = "/home/ro/ML-DL-CV/opencv_code/OpenCV-main/02_Video/00_Scenery.mp4"
vc=cv2.VideoCapture(video_path)
# 检查视频是否打开正常
if vc.isOpened():   # 检查是否打开正确
    open, frame = vc.read() # 这里的 vc.read() 相当于读取图像的第一帧[open会返回布尔值，]
                            # 若循环不断的执行 vc.read，则不断的读取第二帧、第三帧....
    print(open)  # 正常打开时，open会返回 True
else:
    open = False

while open: # 如果正常打开，则遍历每一帧,这里可替换成 i 值，来确定读取 i 帧
    ret, frame = vc.read()
    if frame is None: # 视频读完以后的下一帧为空，则退出遍历
        break
    if ret == True: # 若读取的该帧可行
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 读取的图片转换成黑白的
        cv2.imshow('result',gray)
        if cv2.waitKey(10) & 0xFF == 27: # cv2.waitKey(10)为等多少时间执行下一帧，0xFF为退出键ESC
            break
vc.release() # release()完成与 open() 相反的工作.释放 open() 向内核申请的所有资源
cv2.destroyAllWindows() # 销毁所有窗口