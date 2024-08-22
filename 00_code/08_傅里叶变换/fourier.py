import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('F:/Pycharm/opencv/opencv/OpenCV-main/01_Picture/13_Lena.jpg',0)
img_float = np.float32(img) # opencv执行傅里叶变换操作时要要求输入 np.float32 格式
dft = cv2.dft(img_float, flags = cv2.DFT_COMPLEX_OUTPUT) # 执行傅里叶变换
dft_shift = np.fft.fftshift(dft) # 将低频值，频率为 0 的部分转换到中间的位置

# 得到灰度图能表示的形式
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1])) # 对两个通道进行转换才能得到图像形式表达，由于转换后的值为非常小的数值，因此还要转换到 0-255 之间
plt.subplot(121), plt.imshow(img,cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([]) # 越往中心频率越低(被 shift 拉到中间)，越往两侧频率越高
plt.show()