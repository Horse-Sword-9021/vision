from all import threshold_value
from os import close
import warnings
import cv2  # 图像格式BGR
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import matplotlib.image as gimage
import numpy as np
from numpy.core.numeric import True_
from numpy.lib.function_base import _gradient_dispatcher, median, piecewise

img = cv2.imread('imgs/1.jpg')
img1 = cv2.imread('imgs/1.jpg')  # 彩色
img2 = cv2.imread('imgs/2.jpg')

# cv2.imshow(p_window,'1.jpg')
img_gray = cv2.imread('imgs/2.jpg', cv2.IMREAD_GRAYSCALE)  # 灰度图


# print(img_gray)
# cv2.imwrite('gray.jpg',img_gray)
# print(img)  #矩阵存储
# print(img.shape) #[h,w,c]

def window_():
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    return key


'''
p_name=input()

class img_s:
    original=cv2.imread(p_name)
    gray=cv2.imread(p_name,cv2.IMREAD_GRAYSCALE)

    def save(save_name,img):
        cv2.imwrite(save_name,img)'''


def picture_show(choose, window_name, *p_name):
    def cv_show(item):
        # cv2.imread(p_name)#bgr,0~255
        cv2.imshow(window_name, item)
        # img_gray=cv2.imread('1.jpg',cv2.IMREAD_GRAYSCALE)#灰度图
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        return key
        # print(img)  #矩阵存储
        # print(img.shape) #[h,w,c]

    def plt_show(item):
        # plt.imread(p_name)
        plt.imshow(item, 'gray')
        plt.title(window_name)
        plt.show()
        key=plt.waitforbuttonpress()
        return key

    if choose == 1:
        for item in p_name:
            cv_show(item)
        return cv_show
    elif choose == 2:
        for item in p_name:
            plt_show(item)
        return plt_show
    else:
        print('again')


# picture_show(2,'plt',plts)


'''
mubiao=img[0:50,0:200]#图像截取
#保留b/g/r通道
imgcopy=img.copy()

imgcopy[:,:,1]=0
imgcopy[:,:,2]=0

imgcopy[:,:,0]=0
imgcopy[:,:,2]=0

imgcopy[:,:,0]=0
imgcopy[:,:,1]=0

b,g,r=cv2.split(img)#切分三个通道
img=cv2.merge((b,g,r))#组合
picture_show('c',imgcopy)'''

# 边界填充
# img=plt.imread('1.jpg')
'''
top_size,bottom_size,left_size,right_size=(50,50,50,50)
replicate=cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,borderType=cv2.BORDER_REPLICATE)
reflect=cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,cv2.BORDER_REFLECT)
reflect101=cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,cv2.BORDER_REFLECT_101)
wrap=cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,cv2.BORDER_WRAP)
constant=cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,borderType=cv2.BORDER_CONSTANT,value=0)
plt.subplot(231),plt.imshow(img,'gray'),plt.title('original');plt.waitforbuttonpress(0);
plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('replicate');plt.waitforbuttonpress(0)
plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('reflect');plt.waitforbuttonpress(0)
plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('reflect101');plt.waitforbuttonpress(0)
plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('wrap');plt.waitforbuttonpress(0)
plt.subplot(236),plt.imshow(constant,'gray'),plt.title('constant');plt.waitforbuttonpress(0)
'''

# 数值计算
'''
img1_1=img1[:5,:,0]
img2_2=img2[:5,:,0]
print(img1_1,img2_2)
print(img1_1+img2_2)'''

# 图像融合
'''
img2=cv2.resize(img2,(864,855))
res=cv2.addWeighted(img1,0.4,img2,0.6,0)
#or
#res=cv2.resize(img,(0,0),fx=4,fy=2)#倍数变化#y=a*x1+b*x2+c
plt.imshow(res)
plt.waitforbuttonpress()'''

# 图像阈值，二值处理
'''
ret,thresh1=cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)#阈值，输出图.#超过阈值部分取最大，否则取0
ret,thresh2=cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY_INV)#反转
ret,thresh3=cv2.threshold(img_gray,127,255,cv2.THRESH_TRUNC)#超过阈值部分设为阈值，否则不变
ret,thresh4=cv2.threshold(img_gray,127,255,cv2.THRESH_TOZERO)#超过阈值不变，否则0
ret,thresh5=cv2.threshold(img_gray,127,255,cv2.THRESH_TOZERO_INV)
titles=['orginal image','binary','binary_iny','trunc','tozero','tozero_iyn']
images=[img,thresh1,thresh2,thresh3,thresh4,thresh5]
for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show(),plt.waitforbuttonpress(0)
'''
# 自适应阈值处理
athdMEAN = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 3)
athdGAUSS = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 3)
picture_show(1, 'adapt', athdMEAN, athdGAUSS)
# 平滑处理/降噪
# 均值滤波，简单平均卷积操作
'''blur=cv2.blur(img,(3,3))
#cv2.imshow('blur',blur)
cv2.waitKey(0)
#方框滤波，同上，易越界
box=cv2.boxFilter(img,-1,(3,3),normalize=True)
#cv2.imshow('box',box)
cv2.waitKey(0)
#高斯滤波/更重视中间的
aussian=cv2.GaussianBlur(img,(5,5),1)
#cv2.imshow('aussian',aussian)
cv2.waitKey(0)
#中值滤波、中值代替
median_=cv2.medianBlur(img,5)#中值滤波
#cv2.imshow('median',median_)
cv2.waitKey(0)
#展示所有
res=np.hstack((blur,aussian,median_))#hstack
#print(res)
cv2.imshow('median vs average',res)
cv2.waitKey(0)'''

# 形态学-腐蚀操作
'''kernel=np.ones((5,5),np.uint8)
erosion=cv2.erode(img,kernel,iterations=1)
cv2.imshow('erosion',erosion),cv2.waitKey(0)

pie=cv2.imread('2.jpg')
kernel=np.ones((30,30),np.uint8)
erosion_1=cv2.erode(pie,kernel,iterations=1)
erosion_2=cv2.erode(pie,kernel,iterations=2)
erosion_3=cv2.erode(pie ,kernel,iterations=3)
res=np.hstack((erosion_1,erosion_2,erosion_3))
cv2.imshow('res',res),cv2.waitKey(0)'''

# 形态学-膨胀操作
'''
kernel=np.ones((3,3),np.uint8)
dige_dilate=cv2.imread('2.jpg')
dige_dilate=cv2.dilate(dige_dilate,kernel,iterations=1)
cv2.imshow('dilate',dige_dilate),cv2.waitKey(0)

pie=cv2.imread('2.jpg')
kernel=np.ones((30,30),np.uint8)
dilate1=cv2.dilate(pie,kernel,iterations=1)
dilate2=cv2.dilate(pie,kernel,iterations=2)
dilate3=cv2.dilate(pie,kernel,iterations=3)
res=np.hstack((dilate1,dilate2,dilate3))
cv2.imshow('res',res),cv2.waitKey(0)'''

# 开闭运算
'''
#开，先腐蚀，再膨胀
kernel=np.ones((5,5),np.uint8)
opening=cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
cv2.imshow('opening',opening),window_()

#闭,反开
closing=cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
cv2.imshow('closing',closing),window_()'''
# 礼貌、黑帽
'''
kernel=np.ones((7,7),np.uint8)
tophat=cv2.morphologyEx(img2,cv2.MORPH_TOPHAT,kernel)#(src,op,kernel)
cv2.imshow('tophat',tophat),window_()

blackhat=cv2.morphologyEx(img2,cv2.MORPH_BLACKHAT,kernel)
cv2.imshow('blackhat',blackhat),window_()'''

# 梯度运算,膨胀-腐蚀
'''
pie=cv2.imread('2.jpg')
kernel=np.ones((7,7),np.uint8)
dilate=cv2.dilate(pie,kernel,iterations=5)
erosion=cv2.dilate(pie,kernel,iterations=5)
res=np.hstack((dilate,erosion))
cv2.imshow('res',res),window_()

gradient=cv2.morphologyEx(pie,cv2.MORPH_GRADIENT,kernel)
cv2.imshow('gradient',gradient),window_()'''

# 图像梯度，Sobel算子.边缘检测
"""
#sobelx=cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
#picture_show(1,'solbex',sobelx)
sobelx=cv2.Sobel(img_gray,cv2.CV_64F,1,0,ksize=3)#(src,ddepth,dx,dy,ksize)
sobelx=cv2.convertScaleAbs(sobelx)

sobely=cv2.Sobel(img_gray,cv2.CV_64F,0,1,ksize=3)
sobely=cv2.convertScaleAbs(sobely)
sobelxy=cv2.addWeighted(sobelx,0.5,sobely,0.5,0)
'''
sobelxy=cv2.Sobel(img2,cv2.CV_64F,1,1,ksize=3)
sobelxy=cv2.convertScaleAbs(sobelxy)'''
picture_show(1,'sobelxy',sobelxy)


#Scharr差异明显化
scharrx=cv2.Scharr(img_gray,cv2.CV_64F,1,0)
scharry=cv2.Scharr(img_gray,cv2.CV_64F,0,1)
scharrx=cv2.convertScaleAbs(scharrx)
scharry=cv2.convertScaleAbs(scharry)
scharrxy=cv2.addWeighted(scharrx,0.5,scharry,0.5,0)


#Laplacian，二阶导，更敏感
laplacian=cv2.Laplacian(img_gray,cv2.CV_64F)
laplacian=cv2.convertScaleAbs(laplacian)
res=np.hstack((sobelxy,scharrxy,laplacian))
picture_show(1,'res',res)
"""

# Canny边缘检测
'''
v1=cv2.Canny(img2,80,150)#(img,minval,maxval)
v2=cv2.Canny(img2,80,190)#50,100
#res=np.hstack((v1,v2))
picture_show(1,'res',v2)'''

# 金字塔
# 高斯金字塔
'''
up=cv2.pyrUp(img)
picture_show(1,'up',up)
down=cv2.pyrDown(img)
picture_show(1,'down',down)
#拉普拉斯金字塔---?
down=cv2.pyrDown(img)
down_up=cv2.pyrUp(down)
l_l=img-down_up
picture_show(1,',',l_l)'''

# 图像轮廓(img,mode,method)

gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)  # 二值图像，准确率高
contours, binary = cv2.findContours(thresh, cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_NONE)  # opencv4与3版本区别，2返回值，counters是第一返回值！！！
# 绘制轮廓---
draw_img = img2.copy()
res = cv2.drawContours(draw_img, contours, -1, (0, 0, 255), 1)
# picture_show(1,'res',res)
# 轮廓特征
cnt = contours[0]  #
'''area=cv2.contourArea(cnt)
perimeter=cv2.arcLength(cnt,True)

#轮廓近似
res=cv2.drawContours(draw_img,[cnt],-1,(0,0,255),2)
picture_show(1,'res',res)
epsilon=0.1*cv2.arcLength(cnt,True)
approx=cv2.approxPolyDP(cnt,epsilon,True)'''

# 外接矩形
'''
x,y,w,h=cv2.boundingRect(cnt)
img_=cv2.rectangle(img2,(x,y),(x+w,y+h),(0,255,0),1)#(image, start_point, end_point, color, thickness)

area=cv2.contourArea(cnt)
rect_area=w*h
extent=area/rect_area
#外接圆
(x,y),radius=cv2.minEnclosingCircle(cnt)
center=(int(x),int(y))
radius=int(radius)
cv2.circle(img2,center,radius,(0,255,0),1)
picture_show(1,'img_',img_)'''

# 模板匹配:https://docs.opencv.org/3.3.1/df/dfb/group__imgproc__objcet.html#ga3a7850640f1fe1f58fe91a2d7583695d
'''
template = cv2.imread('1.jpg', 0)

h, w = template.shape[:2]
methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]
res = cv2.matchTemplate(img, template, 1, cv2.TM_SQDIFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

for meth in methods:
    img_ = img2.copy()
    res = cv2.matchTemplate(img_, template, meth)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    if meth in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = max_loc
    else:
        bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img, top_left, bottom_right, 255, 2)
    plt.subplot(121), plt.imshow(res, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img, cmap='gray')
    plt.xticks([]), plt.yticks
    plt.suptitle('{meth}')
    plt.show()

threshold = 0.8
loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):
    bottom_right = (pt[0] + w, pt[1] + h)
    cv2.rectangle(img, pt, bottom_right, (0, 0, 255), 2)

picture_show(1, 'img_rgb', img)'''
