#from code.all import threshold_
from os import close
import warnings
import cv2
from matplotlib.colors import Normalize  # 图像格式BGR
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.numeric import True_
from numpy.lib.function_base import _gradient_dispatcher, median, piecewise
import time
import all





def main():
    engine_img = []
    plt_engine_img = []
    gray_engine_img = []
    smooth_engine_img=[]
    gray_smooth_engine_img=[]
    threshold_engine_img=[]
    erosion_img=[]
    close_img=[]
    for i in range(1, 66):
        engine_img.append(cv2.imread(f'imgs/imgs/{i}.png'))
        plt_engine_img.append(cv2.imread(f'imgs/imgs/{i}.png'))
        gray_engine_img.append(cv2.imread(f'.imgs/imgs/{i}.png', cv2.IMREAD_GRAYSCALE))

        #高斯平滑+二值
        try:
            all.smooth(engine_img[i-1],i)
        except Exception as e:
            print('Check your code:smooth()', e.__class__.__name__, e)  # continue #jia

        smooth_engine_img.append(cv2.imread(f'imgs/operate_imgs/1smooth/{i}_bilateral.png'))
        gray_smooth_engine_img.append(cv2.imread(f'imgs/operate_imgs/1smooth/{i}_bilateral.png',cv2.IMREAD_GRAYSCALE))
        try:
            all.threshold_(smooth_engine_img[i-1],gray_smooth_engine_img[i-1],i)
        except Exception as e:
            print('Check your code:threshold_()', e.__class__.__name__, e)  # continue #jia

        threshold_engine_img.append(cv2.imread(f'imgs/operate_imgs/2thresh_adapt/{i}_adapt.png'))
        try:
            all.morphology(smooth_engine_img[i-1],i-1)
        except Exception as e:
            print('Check your code:morphology()', e.__class__.__name__, e)  # continue #jia

        erosion_img.append(cv2.imread(f'imgs/operate_imgs/3mor_erosion/{i}_erosion.png'))
        try:
            all.open_close_caculate_hat(erosion_img[i-1],i)
        except Exception as e:
            print('Check your code:open_close_caculate_hat()', e.__class__.__name__, e)  # continue #jia

        close_img.append(cv2.imread(f'imgs/operate_imgs/4close/{i}_close.png'))
        try:
            all.picture_gradient_sobel(close_img[i-1],i)
        except Exception as e:
            print('Check your code:picture_gradient_sobel()', e.__class__.__name__, e)  # continue #jia
        try:
            all.canny(close_img[i-1],i)
        except Exception as e:
            print('Check your code:canny()', e.__class__.__name__, e)  # continue #jia
        try:
            all.outline(close_img[i-1],i)
        except Exception as e:
            print('Check your code:outline()', e.__class__.__name__, e)  # continue #jia

        '''
            all.boundary_fill(plt_engine_img[i]),plt.savefig(f'imgs/imgs/{i}.png')'''




if __name__ == '__main__':
    try:
        start=time.time()
        main()
        print(f'耗时{time.time()-start}秒')
    except Exception as e:
        print('Check your code:main()', e.__class__.__name__, e)  # continue#jia    

    '''

    while 1:
        main()
        time.sleep(2)
        '''
