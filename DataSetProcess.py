#import tensorflow as tf
from re import X
from turtle import width
import numpy as np
import glob
import os
from skimage import io, transform, color
import time
import cv2

raw_path='D:/Microelectronics/GraduationProject/MainProject/cnn/DataSet/'
pro_path='D:/Microelectronics/GraduationProject/MainProject/cnn/DataSetProcessed_3/'
width=32
hight=32



cate=[raw_path+x for x in os.listdir(raw_path) if os.path.isdir(raw_path + x) ]
imgs=[]
labels=[]
for index, folder in enumerate(cate):
    i=0
    for img_addr in glob.glob(folder + '\*.jpg'):
        i+=1
        print('reading the images:%s' %(img_addr))
        img=cv2.imread(img_addr, 0)
        ret, thresh = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV)
        thresh=cv2.resize(thresh,(hight,width))
        save_path=pro_path + '%s'%(str(index))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cv2.imwrite(save_path+'/%s.jpg'%(str(i)), thresh)
        print(i)


# frame_vga=cv2.imread(path+'TestPhoto.jpg')
# img_ycrcb = cv2.cvtColor(frame_vga,cv2.COLOR_BGR2YCR_CB)
# (y, cr, cb) = cv2.split(img_ycrcb)
# cr1 = cv2.GaussianBlur(cr, (5, 5), 0)
# _, skin = cv2.threshold(cr,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# _, skin_blur = cv2.threshold(cr1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


# print(np.asarray(skin).size)
# print(np.asarray(frame_vga).size)
# cv2.imshow('first',frame_vga)
# cv2.imshow('second',skin)
# cv2.imshow('third',skin_blur)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# blured=cv2.GaussianBlur(frame_vga,(5,5),0)
# cv2.imshow('first',frame_vga)
# cv2.imshow('second',blured)
# cv2.waitKey(0)
# cv2.destroyAllWindows

# raw_photo=cv2.imread(raw_path+'/1/20one0.jpg',0)
# raw_photo=cv2.imread('C:/Users/28155/Desktop/hap.jpg',0)
# 直接读取为灰度图片
# print(raw_photo)
# gray_image = cv2.cvtColor(raw_photo, cv2.COLOR_BGR2GRAY)
# print(gray_image)
# data=np.asarray(gray_image).size

# ret, thresh = cv2.threshold(raw_photo, 150, 255, cv2.THRESH_BINARY_INV)
# thresh=cv2.resize(thresh,(50,50))
# cv2.imshow('source',raw_photo)
# cv2.imshow('result',thresh)
# cv2.imwrite('C:/Users/28155/Desktop/20one0.jpg',thresh)
# cv2.waitKey(0)
# cv2.destroyAllWindows

