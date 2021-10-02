'''步骤简述：先对图片进行均值模糊是因为原图中有类似波浪状的干扰，想先模糊图片；然后使用拉普拉斯算子得到原图的一些局部特征，将它和原图进行相加减，便可使得原图中这些特征更加明显；然后使用自适应阈值分割，并在原图上绘制所得的框。'''

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

src = cv.imread("图片地址", cv.IMREAD_GRAYSCALE)
#均值滤波
src_blur = cv.blur(src, (3, 3))
#拉普拉斯锐化
lapmask = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], np.float32)
src_lap = cv.filter2D(src_blur, -1, kernel=lapmask)
#对拉普拉斯图阈值分割（二值分割）
ret, th = cv.threshold(src_lap, 50, 255, cv.THRESH_BINARY)
#用模糊后的原图减去其拉普拉斯图，使得缺陷部位更清晰
img = src_blur - abs(50*th)
#自适应阈值分割
dst = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 5, 5)
#轮廓提取及绘制
binary, contours, hierarchy = cv.findContours(dst, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
src_rgb = cv.cvtColor(src, cv.COLOR_GRAY2BGR)
cv.drawContours(src_rgb, contours, -1, (255, 0, 0), 1)
cv.imwrite("保存地址", src_rgb)

cv.namedWindow("test", cv.WINDOW_AUTOSIZE)
cv.imshow("test", src_rgb)
plt.hist(src_rgb.ravel(), 256, [0, 256])
plt.show("直方图")

cv.waitKey(0)
cv.destroyAllWindows()

