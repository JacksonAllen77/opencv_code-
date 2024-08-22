import cv2 #opencv的缩写为cv2
import matplotlib.pyplot as plt # matplotlib库用于绘图展示
import numpy as np   # numpy数值计算工具包
def cv_show(name, img):
    # 显示图像
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  # 使用 destroyAllWindows() 关闭所有窗口

img = cv2.imread('/home/ro/ML-DL-CV/opencv_code/OpenCV-main/01_Picture/05_Dige.png')
cv_show("origin",img)

kernel = np.ones((5,5),np.uint8) # 创建一个所有元素都为1的5×5的卷积核，数据类型为无符号 8 位整数（即 0 到 255 的整数）
erosion = cv2.erode(img,kernel,iterations=1) # 对原图运用卷积核执行腐蚀操作，迭代次数为1【可以增加迭代次数进一步腐蚀】
cv_show("ones(5×5)",erosion)

# 不同卷积核的结果展示
kernel2 = np.ones((2,2),np.uint8) # 创建一个所有元素都为1的5×5的卷积核，数据类型为无符号 8 位整数（即 0 到 255 的整数）
erosion2 = cv2.erode(img,kernel2,iterations=1) # 对原图运用卷积核执行腐蚀操作，迭代次数为1【可以增加迭代次数进一步腐蚀】
cv_show("ones(2×2)",erosion2)

# 不同迭代次数的结果展示
kernel3 = np.ones((5,5),np.uint8) # 创建一个所有元素都为1的5×5的卷积核，数据类型为无符号 8 位整数（即 0 到 255 的整数）
erosion3 = cv2.erode(img,kernel3,iterations=2) # 对原图运用卷积核执行腐蚀操作，迭代次数为1【可以增加迭代次数进一步腐蚀】
cv_show("twice(5×5)",erosion3)