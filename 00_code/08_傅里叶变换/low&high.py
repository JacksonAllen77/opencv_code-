import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('F:/Pycharm/opencv/opencv/OpenCV-main/01_Picture/13_Lena.jpg',0)# 读取灰度图像
img_float32 = np.float32(img) # 转换为 float32 类型【傅里叶变换的必要准备】

# DFT ( 傅里叶变换 )
dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft) # 将低频部分转移到中心位置

rows, cols = img.shape
crow, ccol = int(rows/2), int(cols/2) # 中心位置

# 低通滤波
mask = np.zeros((rows, cols, 2), np.uint8)  # 创建一个与图像大小相同的掩码，初始化为零
mask[crow-30:crow+30, ccol-30:ccol+30] = 1  # 在掩码中心区域设置为1，保留低频部分

# IDPT (傅里叶逆变换)
fshift = dft_shift * mask # 用掩码提取 dft_shift 中相应区域，是 1 就保留，不是 1 就过滤了
f_ishift = np.fft.ifftshift(fshift) # 把拉到中心位置的频谱区域给还原回去，依旧回到左上角
img_back = cv2.idft(f_ishift) # 进行傅里叶逆变换，将图像从频域还原到时域
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1]) # 将实部和虚部结合起来，才能将傅里叶变换的结果显示出来

plt.subplot(121),plt.imshow(img,cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_back,cmap='gray')
plt.title('Low'), plt.xticks([]), plt.yticks([])
plt.show()

