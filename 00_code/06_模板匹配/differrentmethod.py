import cv2 #opencv的缩写为cv2
import matplotlib.pyplot as plt # matplotlib库用于绘图展示
import numpy as np   # numpy数值计算工具包

template=cv2.imread("/home/ro/ML-DL-CV/opencv_code/OpenCV-main/01_Picture/12_Face.jpg",0)
img=cv2.imread("/home/ro/ML-DL-CV/opencv_code/OpenCV-main/01_Picture/13_Lena.jpg",0)
h, w = template.shape[:2] # 获得模板的宽和高
methods = ['cv2.TM_CCOEFF','cv2.TM_CCOEFF_NORMED','cv2.TM_CCORR',
          'cv2.TM_CCORR_NORMED','cv2.TM_SQDIFF','cv2.TM_SQDIFF_NORMED']

for meth in methods:
    img2 = img.copy()
    # 匹配方法的真值
    method = eval(meth) # 提取字符串中的内容，不能用字符串的形式
    print(method)
    res = cv2.matchTemplate(img ,template ,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # 如果是平方差匹配 TM_SQDIFF 或归一化平方差匹配 TM_SQDIFF_NORMED,取最小值
    if method in [cv2.TM_SQDIFF ,cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0 ] +w ,top_left[1 ] +h)

    # 画矩形
    cv2.rectangle(img2 ,top_left ,bottom_right ,255 ,2)

    plt.subplot(121), plt.imshow(res, cmap='gray')
    plt.xticks([]), plt.yticks([]) # 隐藏坐标轴
    plt.subplot(122) ,plt.imshow(img2 ,cmap='gray')
    plt.xticks([]) ,plt.yticks([])
    plt.suptitle(meth)
    plt.show()