import cv2 
import os
import glob

DataSet='C:/Users/28155/Desktop/github/test/'
DataSet_processed='C:/Users/28155/Desktop/github/test_processed/'

cate=[DataSet+x for x in os.listdir(DataSet) if os.path.isdir(DataSet + x) ]

for index, folder in enumerate(cate):
    i=0
    for img_addr in glob.glob(folder + '\*.jpg'):
        i+=1
        print('reading the images:%s' %(img_addr))
        img=cv2.imread(img_addr)
        img=img[:,80:560:]
        img_ycrcb = cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
        (y, cr, cb) = cv2.split(img_ycrcb)
        cr1 = cv2.GaussianBlur(cr, (5, 5), 0)
        _, skin_out = cv2.threshold(cr1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        skin_out=cv2.resize(skin_out,(32,32))
        save_path=DataSet_processed + '%s'%(str(index))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cv2.imwrite(save_path+'/%s.jpg'%(str(i)), skin_out)

        