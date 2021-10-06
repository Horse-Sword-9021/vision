import cv2  #图像格式BGR
import numpy as np
import matplotlib.pyplot as plt

def video_show(v_name):
    vc=cv2.VideoCapture(v_name)
    if vc.isOpened():#是否正确打开
        open,frame=vc.read()
    else:
        open=False

    while open:
        ret,frame=vc.read()
        if frame is None:
            break
        if ret==True:
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#转换灰度
            cv2.imshow('result',gray)
            if cv2.waitKey(10)&0xFF==27: #参数-帧率
                break
    vc.release()
    key=cv2.waitKey(0)
    cv2.destroyAllWindows()

video_show('imgs/1.mp4')
