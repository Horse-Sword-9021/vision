import cv2  # 图像格式BGR
from matplotlib.colors import Normalize
import numpy as np
import matplotlib.pyplot as plt



def window_():
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    return key


def picture_show(choose, window_name,*p_name):

    def cv_show(window_name,item):
        methods_imread = [cv2.IMREAD_COLOR, cv2.IMREAD_GRAYSCALE]
        # cv2.imread(p_name)#bgr,0~255
        cv2.imshow(window_name, item)
        # img_gray=cv2.imread('1.jpg',cv2.IMREAD_GRAYSCALE)#灰度图
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        return key
        # print(img)  #矩阵存储
        # print(img.shape) #[h,w,c]

    def plt_show(window_name,item):
        # plt.imread(p_name)
        plt.imshow(item, 'gray')
        plt.title(window_name)
        plt.show()
        # key=plt.waitforbuttonpress()
        # return key

    if choose == 1:
        for item in p_name:
            cv_show(window_name,item)
        return cv_show
    elif choose == 2:
        for item in p_name:
            plt_show(window_name, p_name)
        return plt_show
    else:
        print('again')


def pyramid(img):  # 金字塔
    def gauss_pyrmaid(img):  # 高斯金字塔
        up = cv2.pyrUp(img)
        picture_show(1, 'up', up)
        down = cv2.pyrDown(img)
        picture_show(1, 'down', down)

    def lpls_pyrmaid(img):  # 拉普拉斯金字塔---?
        down = cv2.pyrDown(img)
        down_up = cv2.pyrUp(down)
        l_l = img - down_up
        picture_show(1, ',', l_l)


def outline(img):  # 轮廓操作
    draw_img = img.copy()
    methods_cvtcolor = [cv2.COLOR_BGR2RGB, cv2.COLOR_BGR2GRAY]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)  # 二值图像，准确率高
    contours, binary = cv2.findContours(thresh, cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_NONE)  # (img,mode,method)#opencv4与3版本区别，2返回值，counters是第一返回值！！！
    cnt = contours[0]

    def draw_outline(img):  # 绘制轮廓---
        res = cv2.drawContours(draw_img, contours, -1, (0, 0, 255), 1)
        picture_show(1, 'res', res)

    def outline():  # 轮廓特征
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        return (area, perimeter)

    def ounline_approximate(args):  # 轮廓近似
        # res=cv2.drawContours(draw_img,[cnt],-1,(0,0,255),2)
        # picture_show(1,'res',res)
        epsilon = 0.1 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        res = cv2.drawContours(draw_img, [approx], -1, (0, 0, 255), 2)
        picture_show(1, 'res', res)

    def rectangle(img):  # 外接矩形
        x, y, w, h = cv2.boundingRect(cnt)
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0),
                            1)  # (image, start_point, end_point, color, thickness)
        area = cv2.contourArea(cnt)
        rect_area = w * h
        extent = area / rect_area  # 轮廓面积与矩形比
        picture_show(1, 'rectangle', img)

    def circle(img):  # 外接圆
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(img, center, radius, (0, 255, 0), 1)
        picture_show(1, 'circle', img)


def mode_match(img):  # 模板匹配
    template = cv2.imread('1.jpg', 0)
    h, w = template.shape[:2]
    methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF,
               cv2.TM_SQDIFF_NORMED]
    res = cv2.matchTemplate(img, template, 1, cv2.TM_SQDIFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    for meth in methods:
        img_ = img.copy()
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

    picture_show(1, 'img_rgb', img)