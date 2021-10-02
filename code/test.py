import all
from os import close
import warnings
import cv2  # 图像格式BGR
from matplotlib.colors import Normalize
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.numeric import True_
from numpy.lib.function_base import _gradient_dispatcher, median, piecewise
import time

'''
engine_img = []
plt_engine_img = []
gray_engine_img = []
for i in range(1, 66):
    try:
        engine_img.append(cv2.imread(f'imgs/imgs/{i}.png'))
        plt_engine_img.append(plt.imread(f'imgs/imgs/{i}.png'))
        #gray_engine_img.append(cv2.imread(f'imgs/imgs/{i}.png', cv2.IMREAD_GRAYSCALE))
    except Exception as e:
        print('Check your code:', e.__class__.__name__, e)  # continue#jia
        continue'''
plt_engine_img=plt.imread('imgs/imgs/1.png')
engine_img=cv2.imread('imgs/imgs/5.png')
gray_engine_img=cv2.imread(f'imgs/imgs/1.png', cv2.IMREAD_GRAYSCALE)

#all.boundary_fill(plt_engine_img)


#all.smooth(engine_img,1)

smooth_engine_img=cv2.imread('imgs/operate_imgs/1smooth/1_bilateral.png')
gray_smooth_engine_img=cv2.imread('imgs/operate_imgs/1smooth/1_bilateral.png',cv2.IMREAD_GRAYSCALE)
#all.threshold_(smooth_engine_img,gray_engine_img,1)
thresh_img=cv2.imread(f'imgs/operate_imgs/thresh_adapt/{1}_adapt.png')
#all.morphology(thresh_img,2)
erosion_img=cv2.imread(f'imgs/operate_imgs/3mor_erosion/{1}_erosion.png')
#all.open_close_caculate_hat(erosion_img,1)

close_img=cv2.imread(f'imgs/operate_imgs/4close/{2}_close.png')

#all.picture_gradient_sobel(close_img)
#all.canny(close_img)
#all.outline(close_img)


#all.histogram(gray_smooth_engine_img)


'''
all.boundary_fill(plt_engine_img[i]),plt.savefig(f'imgs/imgs/{i}.png')'''

def main():
    engine_img = []
    plt_engine_img = []
    gray_engine_img = []
    for i in range(1, 66):
        try:
            engine_img.append(cv2.imread(f'imgs/imgs/{i}.png'))
            plt_engine_img.append(cv2.imread(f'imgs/imgs/{i}.png'))
            gray_engine_img.append(cv2.imread(f'.imgs/imgs/{i}.png', cv2.IMREAD_GRAYSCALE))
        except Exception as e:
            print('Check your code:27', e.__class__.__name__, e)  # continue #jia
        '''
        for i in range(0,65):
            all.boundary_fill(plt_engine_img[i]),plt.savefig(f'imgs/imgs/{i}.png')'''

    #高斯平滑+二值
    for i in range(0,65):
        all.smooth(engine_img[i],i)


    smooth_engine_img=[]
    gray_smooth_engine_img=[]
    for i in range(1,66):
        try:
            smooth_engine_img.append(cv2.imread(f'imgs/operate_imgs/1smooth/{i}_bilateral.png'))
            gray_smooth_engine_img.append(cv2.imread(f'imgs/operate_imgs/1smooth/{i}_bilateral.png',cv2.IMREAD_GRAYSCALE))
        except Exception as e:
            print('Check your code:44', e.__class__.__name__, e)
    
    for i in range(0,65):
        all.morphology(smooth_engine_img[i],i)


    for i in range(0,65):
        all.threshold_(smooth_engine_img[i],gray_smooth_engine_img[i],i)
    threshold_engine_img=[]
    for i in range(1,66):
        try:
            threshold_engine_img.append(cv2.imread(f'imgs/operate_imgs/2thresh_adapt/{i}_adapt.png'))
        except Exception as e:
            print('Check your code:57', e.__class__.__name__, e)
    
    erosion_img=[]
    for i in range(1,66):
        erosion_img.append(cv2.imread(f'imgs/operate_imgs/3mor_erosion/{i}_erosion.png'))
    for i in range(0,65):
        all.open_close_caculate_hat(erosion_img[i],i)
    

    close_img=[]
    for i in range(1,66):
        close_img.append(cv2.imread(f'imgs/operate_imgs/4close/{i}_close.png'))
    
    for i in range(0,65):
        all.picture_gradient_sobel(close_img[i],i)
        all.canny(close_img[i],i)

if __name__ == '__main__':
    try:
        start=time.time()
        main()
        print(f'耗时{time.time()-start}秒')
    except Exception as e:
        print('Check your code:main()', e.__class__.__name__, e)  # continue#jia
    