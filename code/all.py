from os import close
import warnings
import cv2  # 图像格式BGR
from matplotlib.colors import Normalize
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.numeric import True_
from numpy.lib.function_base import _gradient_dispatcher, median, piecewise
import time


def window_():
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    return key

#cv2.imwrite('name',img)

###################################################
class img_s:
    original=cv2.imread('p_name')
    gray=cv2.imread('p_name',cv2.IMREAD_GRAYSCALE)

    def save(save_name,img):
        cv2.imwrite(save_name,img)
###################################################

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
        plt.show(),plt.close()
        #key=plt.waitforbuttonpress()

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


def single_channel(img):  # 单通道保留
    # mubiao=img[0:50,0:200]#图像截取
    # 保留b/g/r通道
    imgcopy = img.copy()
    # b
    imgcopy[:, :, 1] = 0
    imgcopy[:, :, 2] = 0
    # g
    imgcopy[:, :, 0] = 0
    imgcopy[:, :, 2] = 0
    # r
    imgcopy[:, :, 0] = 0
    imgcopy[:, :, 1] = 0

    b, g, r = cv2.split(img)  # 切分三通道
    img = cv2.merge((b, g, r))  # 组合
    picture_show('c', imgcopy)


#good
def boundary_fill(pltimg):  # 边界填充
    #img = plt.imread(str(p_name))
    top_size, bottom_size, left_size, right_size = (50, 50, 50, 50)
    '''
    replicate = cv2.copyMakeBorder(pltimg, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REPLICATE)
    picture_show(2,'replicate',replicate)
    reflect = cv2.copyMakeBorder(pltimg, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REFLECT)
    picture_show(2,'reflect',reflect)
    reflect101 = cv2.copyMakeBorder(pltimg, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REFLECT_101)
    picture_show(2,'reflect101',reflect101)
    wrap = cv2.copyMakeBorder(pltimg, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_WRAP)
    picture_show(2,'wrap',wrap)'''
    constant = cv2.copyMakeBorder(pltimg, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_CONSTANT,
                                  value=0)
    picture_show(2,'constant',constant)
    #cv2.imwrite(f'imgs/operate_imgs/{1}_o_.png',constant)
    
    '''
    imgs = [pltimg, replicate, reflect, reflect101, wrap, constant]
    titles = ['original', 'replicate', 'reflect', 'reflect101', 'wrap', 'constant']
    for i in range(231, 237):
        plt.subplot(i), plt.imshow(imgs[i - 231], 'gray')#, plt.title(titles[i - 231])
    picture_show(2, f' {titles[i-231]}', imgs)
    
    plt.subplot(231), plt.imshow(pltimg, 'gray'), plt.title('original'),#plt.show(),plt.waitforbuttonpress(0)
    plt.subplot(232), plt.imshow(replicate, 'gray'), plt.title('replicate'),plt.show(),plt.waitforbuttonpress(0)
    plt.subplot(233), plt.imshow(reflect, 'gray'), plt.title('reflect'),plt.show(),plt.waitforbuttonpress()
    plt.subplot(234), plt.imshow(reflect101, 'gray'), plt.title('reflect101'),plt.show(),plt.waitforbuttonpress()
    plt.subplot(235), plt.imshow(wrap, 'gray'), plt.title('wrap'),plt.show(),plt.waitforbuttonpress()
    plt.subplot(236), plt.imshow(constant, 'gray'), plt.title('constant'),plt.show(),plt.waitforbuttonpress()
    plt.show()'''


#good
def num_caculate(img):  # 数值计算
    img1_1 = img+10
    img[:5, :, 0]
    img1_1[:5, :, 0]
    #print(img1_1, img2_2)
    print(img1_1 + img)


#good
def picture_merge(img1, img2):  # 图像融合
    img2 = cv2.resize(img2, (864, 855))
    res = cv2.addWeighted(img1, 0.4, img2, 0.6, 0)
    # or
    # res=cv2.resize(img,(0,0),fx=4,fy=2)#倍数变化#y=a*x1+b*x2+c
    picture_show(2, 'res', res)


# good
def threshold_(img, img_gray, i):
    def threshlod():  # 图像阈值，二值处理
        ret, thresh1 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)  # 阈值，输出图.#超过阈值部分取最大，否则取0
        ret, thresh2 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)  # 反转
        ret, thresh3 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TRUNC)  # 超过阈值部分设为阈值，否则不变
        ret, thresh4 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO)  # 超过阈值不变，否则0
        ret, thresh5 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO_INV)
        # Otsu阈值处理
        ret, otsu = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        titles = ['orginal', 'binary', 'binary_iny', 'trunc', 'tozero', 'tozero_iyn', 'otsu']
        images = [img, thresh1, thresh2, thresh3, thresh4, thresh5, otsu]
        for i in range(6):
            plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.show(),plt.close()
        #picture_show(1, 'otsu', otsu)

    def adapt():  # 自适应阈值处理
        #athdmean = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 3)
        athdgauss = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 3)
        cv2.imwrite(f'imgs/operate_imgs/2thresh_adapt/{i}_adapt.png',athdgauss)
        #res=np.hstack((athdmean,athdgauss))
        #picture_show(1, 'adapt', athdmean, athdgauss)
    #threshlod()
    adapt()


#good   +++
def smooth(img,i):  # 平滑处理/降噪
    '''
    # 均值滤波，简单平均卷积操作
    blur = cv2.blur(img, (3, 3))  # (img,ksize)
    picture_show(1, 'blur', blur)
    # 方框滤波，同上，易越界
    box = cv2.boxFilter(img, -1, (3, 3), normalize=True)  # 平均操作
    picture_show(1, 'box', box)
    # 高斯滤波/更重视中间的
    aussian = cv2.GaussianBlur(img, (5, 5), 1)  # (img,ksize,sigmax,sigmay)
    picture_show(1, 'aussian', aussian)
    # 2D卷积
    kernel = np.ones((9, 9), np.float32) / 81
    filter2d = cv2.filter2D(img, -1, kernel)  # (img,-1,kernel)
    picture_show(1, 'filter2d', filter2d)'''
    # 中值滤波、中值代替
    median_ = cv2.medianBlur(img, 5)  # (img,k)
    #picture_show(1, 'median', median_)
    # 双边滤波
    bilateral = cv2.bilateralFilter(img, 25, 100, 100)  # 高斯双边模糊 (0,100,10)
    #picture_show(1, 'bilateral', bilateral)

    # 展示所有
    #res = np.hstack((blur, aussian, median_, bilateral, filter2d))  # hstack
    # print(res)
    #picture_show(1, 'median vs average', res)
    cv2.imwrite(f'imgs/operate_imgs/1smooth/{i}_median.png',median_)
    cv2.imwrite(f'imgs/operate_imgs/1smooth/{i}_bilateral.png',bilateral)

#good
def morphology(img,i):  # 形态学
    kernel = np.ones((3, 3), np.uint8)
    kernel_ = np.ones((30, 30), np.uint8)
    #pie = cv2.imread('imgs/2.jpg')

    def erosion_():  # 形态学-腐蚀操作
        erosion = cv2.erode(img, kernel, iterations=1)  # (src, kernel, iterations)
        #picture_show(1, 'erosion', erosion)
        cv2.imwrite(f'imgs/operate_imgs/3mor_erosion/{i}_erosion.png',erosion)
        '''
        erosion_1 = cv2.erode(img, kernel_, iterations=1)
        erosion_2 = cv2.erode(img, kernel_, iterations=2)
        erosion_3 = cv2.erode(img, kernel_, iterations=3)
        res = np.hstack((erosion_1, erosion_2, erosion_3))
        picture_show(1, 'res', res)'''

    def expend():  # 膨胀？？？
        #dige_dilate = cv2.imread('imgs/2.jpg')
        dige_dilate = cv2.dilate(img, kernel, iterations=1)
        picture_show(1, 'dilate', dige_dilate)
        '''
        dilate1 = cv2.dilate(img, kernel_, iterations=1)
        dilate2 = cv2.dilate(img, kernel_, iterations=2)
        dilate3 = cv2.dilate(img, kernel_, iterations=3)
        res = np.hstack((dilate1, dilate2, dilate3))
        picture_show(1, 'res', res)'''
    erosion_()
    #expend()


#good
def open_close_caculate_hat(img,i):  # 开闭运算
    kernel = np.ones((5, 5), np.uint8)

    def open():  # 开，先腐蚀，再膨胀
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # (src,op,kernel)
        picture_show(1, 'opening', opening)

    def close():  # 闭,先膨胀，再腐蚀
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        #picture_show(1, 'closing', closing)
        cv2.imwrite(f'imgs/operate_imgs/4close/{i}_close.png',closing)

    def top_black_hat():  # 顶帽、黑帽
        kernel = np.ones((7, 7), np.uint8)

        def top():  # 顶帽：原输入-开运算
            tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
            picture_show(1, 'tophat', tophat)

        def black():  # 黑帽：闭运算-原输入
            blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
            picture_show(1, 'blackhat', blackhat)
        top()
        black()

    def gradient():  # 梯度运算,膨胀-腐蚀
        kernel = np.ones((7, 7), np.uint8)
        gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
        picture_show(1, 'gradient', gradient)
    #open()
    close()
    #top_black_hat()
    #gradient()
    
    


#good
def picture_gradient_sobel(img, i):
    def sobel():  # 图像梯度，Sobel算子.边缘检测
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # (src,ddepth,dx,dy,ksize)
        sobelx = cv2.convertScaleAbs(sobelx)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        sobely = cv2.convertScaleAbs(sobely)
        sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
        '''
        sobelxy=cv2.Sobel(img2,cv2.CV_64F,1,1,ksize=3)
        sobelxy=cv2.convertScaleAbs(sobelxy)'''
        # picture_show(1,'sobelxy',sobelxy)
        return sobelxy

    def scharr():  # Scharr差异明显化
        scharrx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
        scharry = cv2.Scharr(img, cv2.CV_64F, 0, 1)
        scharrx = cv2.convertScaleAbs(scharrx)
        scharry = cv2.convertScaleAbs(scharry)
        scharrxy = cv2.addWeighted(scharrx, 0.5, scharry, 0.5, 0)
        # picture_show(1,'scharrxy',scharrxy)
        return scharrxy

    def laplacian():  # Laplacian，二阶导，更敏感
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        laplacian = cv2.convertScaleAbs(laplacian)
        # picture_show(1,'laplacian',laplacian)
        return laplacian

    res = np.hstack((sobel(), scharr(), laplacian()))
    cv2.imwrite(f'imgs/operate_imgs/5sobel_canny/{i}_sobel.png',res)
    #picture_show(1, 'res', res)


#good
def canny(img, i):  # Canny边缘检测
    v1 = cv2.Canny(img, 80, 150)  # (img,minval,maxval)
    v2 = cv2.Canny(img, 50, 100)
    res = np.hstack((v1, v2))
    cv2.imwrite(f'imgs/operate_imgs/5sobel_canny/{i}_canny.png',res)
    #picture_show(1, 'res', res)


#lpls,???
def pyramid(img):  # 金字塔
    def gauss_pyrmaid():  # 高斯金字塔
        up = cv2.pyrUp(img)
        down = cv2.pyrDown(img)
        picture_show(1, 'down', up, down)

    def lpls_pyrmaid():  # 拉普拉斯金字塔---?
        down = cv2.pyrDown(img)
        down_up = cv2.pyrUp(down)
        l_l = img - down_up
        picture_show(1, ',', l_l)
    gauss_pyrmaid(), lpls_pyrmaid()


#good
def outline(img,i):  # 轮廓操作
    draw_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)  # 二值图像，准确率高
    contours, binary = cv2.findContours(thresh, cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_NONE)  # (img,mode,method)#opencv4与3版本区别，2返回值，counters是第一返回值！！！
    cnt = contours[0]

    def draw_outline():  # 绘制轮廓---
        res = cv2.drawContours(draw_img, contours, -1, (0, 0, 255), 1)
        #picture_show(1, 'res', res)
        cv2.imwrite(f'imgs/operate_imgs/5outline/{i}_outline.png',res)

    def outline_feature():  # 轮廓特征
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        return (area, perimeter)

    def outline_approximate():  # 轮廓近似
        # res=cv2.drawContours(draw_img,[cnt],-1,(0,0,255),2)
        # picture_show(1,'res',res)
        epsilon = 0.1 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        res = cv2.drawContours(draw_img, [approx], -1, (0, 0, 255), 2)
        picture_show(1, 'res', res)

    def rectangle():  # 外接矩形
        x, y, w, h = cv2.boundingRect(cnt)
        img_ = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0),
                            1)  # (image, start_point, end_point, color, thickness)
        area = cv2.contourArea(cnt)
        rect_area = w * h
        extent = area / rect_area  # 轮廓面积与矩形比
        picture_show(1, 'rectangle', img_)

    def circle():  # 外接圆
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(img, center, radius, (0, 255, 0), 1)
        picture_show(1, 'circle', img)
    #circle()
    #rectangle()
    #outline_approximate()#
    #print(outline_feature())
    draw_outline()


#========================================================================================================


def mode_match(img):  # 模板匹配
    template = cv2.imread('imgs/1.jpg', 0)
    h, w = template.shape[:2]
    methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF,
               cv2.TM_SQDIFF_NORMED]
    template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
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
#################################################################################################


#good
def histogram(gray_p):  # 直方图均衡化
    #img = cv2.imread(p_name, cv2.IMREAD_GRAYSCALE)
    result = cv2.equalizeHist(gray_p)
    picture_show(1, 'equal', result)


def rotate():
    src = cv2.imread("D:/images/dannis1.png")
    cv2.imshow("input", src)
    h, w, c = src.shape
    M = np.zeros((2, 3), dtype=np.float32)
    alpha = np.cos(np.pi / 4.0)
    beta = np.sin(np.pi / 4.0)
    print("alpha : ", alpha)

    # 初始旋转矩阵
    M[0, 0] = alpha
    M[1, 1] = alpha
    M[0, 1] = beta
    M[1, 0] = -beta
    cx = w / 2
    cy = h / 2
    tx = (1 - alpha) * cx - beta * cy
    ty = beta * cx + (1 - alpha) * cy
    M[0, 2] = tx
    M[1, 2] = ty

    # change with full size
    bound_w = int(h * np.abs(beta) + w * np.abs(alpha))
    bound_h = int(h * np.abs(alpha) + w * np.abs(beta))

    # 添加中心位置迁移
    M[0, 2] += bound_w / 2 - cx
    M[1, 2] += bound_h / 2 - cy
    dst = cv2.warpAffine(src, M, (bound_w, bound_h))
    cv2.imshow("rotate without cropping", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def image_hist():
    image = cv2.imread("imgs/2.jpg")
    cv2.imshow("input", image)
    color = ('blue', 'green', 'red')
    for i, color in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [32], [0, 256])
        print(hist)
        plt.plot(hist, color=color)
        plt.xlim([0, 32])
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def hist2d(img):  # 2d直方图
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [48, 48], [0, 180, 0, 256])
    dst = cv2.resize(hist, (400, 400))
    cv2.normalize(dst, dst, 0, 255, cv2.NORM_MINMAX)  # 归一化
    # cv2.imshow("image", img)
    dst = cv2.applyColorMap(np.uint8(dst), cv2.COLORMAP_JET)  # 填充
    picture_show(1, 'hist', img, dst, hist)
    # picture_show(1,'.',hist)


def video(v_name):
    def video_demo(v_name):
        cap = cv2.VideoCapture(v_name)
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter("D:/test.mp4", cv2.CAP_ANY, np.int(cap.get(cv2.CAP_PROP_FOURCC)), fps,
                              (np.int(w), np.int(h)), True)
        print(w, h, fps)
        while True:
            ret, frame = cap.read()
            if ret is not True:
                break
            cv2.imshow("frame", frame)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            cv2.imshow("result", hsv)
            out.write(hsv)
            c = cv2.waitKey(10)
            if c == 27:
                break
        cv2.destroyAllWindows()

        out.release()
        cap.release()

    def video_show(v_name):
        vc = cv2.VideoCapture(str(v_name))
        if vc.isOpened():  # 是否正确打开
            open_, frame = vc.read()
        else:
            open_ = False

        while open_:
            ret, frame = vc.read()
            if frame is None:
                break
            if ret == True:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转换灰度
                cv2.imshow('result', gray)
                if cv2.waitKey(10) & 0xFF == 27:  # 参数
                    break
        vc.release()
        key = window_()
        return key


def test():
    model_bin = 'code/opencv_detector_uint8.pb'
    config_text = 'code/opencv_detector.pbtxt'

    def face_detection():
        net = cv2.dnn.readNetFromTensorflow(model=model_bin, config=config_text)
        cap = cv2.VideoCapture("imgs/face.mp4")
        while True:
            ret, frame = cap.read()
            h, w, c = frame.shape
            if ret is not True:
                break
            # NCHW
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
            net.setInput(blob)
            outs = net.forward()  # 1x1xNx7
            for detection in outs[0, 0, :, :]:
                score = float(detection[2])
                if score > 0.5:
                    left = detection[3] * w
                    top = detection[4] * h
                    right = detection[5] * w
                    bottom = detection[6] * h
                    cv2.rectangle(frame, (np.int(left), np.int(top)), (np.int(right), np.int(bottom)), (0, 0, 255), 2,
                                  8, 0)
            cv2.imshow("frame", frame)
            c = cv2.waitKey(1)
            if c == 27:
                break
        cv2.destroyAllWindows()
        cap.release()

    def video_detection():
        # load tensorflow model
        net = cv2.dnn.readNetFromTensorflow(model_bin, config=config_text)
        capture = cv2.VideoCapture("imgs/face.mp4")

        # 目标检测
        while True:
            e1 = cv2.getTickCount()
            ret, frame = capture.read()
            if ret is not True:
                break
            h, w, c = frame.shape
            blobImage = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
            net.setInput(blobImage)
            cvOut = net.forward()

            # Put efficiency information.
            t, _ = net.getPerfProfile()
            label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())

            # 绘制检测矩形
            for detection in cvOut[0, 0, :, :]:
                score = float(detection[2])
                objindex = int(detection[1])
                if score > 0.5:
                    left = detection[3] * w
                    top = detection[4] * h
                    right = detection[5] * w
                    bottom = detection[6] * h

                    # 绘制
                    cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), thickness=2)
                    cv2.putText(frame, "score:%.2f" % score, (int(left), int(top)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255), 1)
            e2 = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (e2 - e1)
            cv2.putText(frame, label + (" FPS: %.2f" % fps), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.imshow('face-detection-demo', frame)
            c = cv2.waitKey(1)
            if c == 27:
                break
        cv2.destroyAllWindows()
        cv2.release()


def main():
    p_name = input()

    class img_s:
        original = cv2.imread(p_name)
        gray = cv2.imread(p_name, cv2.IMREAD_GRAYSCALE)

        def save(save_name, img):
            cv2.imwrite(save_name, img)

    a = input()


if __name__ == '__main__':
    main()
    time.sleep(2)
    '''while 1:
        main()
        '''
