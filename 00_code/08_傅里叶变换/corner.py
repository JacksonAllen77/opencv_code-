import cv2
import numpy as np

img = cv2.imread('F:/Pycharm/opencv/opencv/OpenCV-main/01_Picture/17_Chessboard.jpg')
print('img.shape:',img.shape)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04) # 每个点与对应点的相似性地值，即变化值
print('dst.shape:',dst.shape)

img[dst>0.01*dst.max()] = [0,0,255] # 比相似性最大值的百分之一要大，则标注为角点
cv2.imshow('dst',img)
cv2.waitKey(0)
cv2.destroyAllWindows()