import cv2 as cv
import numpy as np

photo_folder='C:/Users/28155/Desktop/1/'
save_folder='C:/Users/28155/Desktop/2/'
for i in range(0,10):
    photo_addr=photo_folder+str(i)+'.jpg'
    img=cv.imread(photo_addr)
    img=cv.resize(img,(68,68))
    mask0 = np.array(img) 
    bool_30=mask0>=127
    mask0[bool_30]=255
    bool_30_=mask0<127
    mask0[bool_30_]=0 
    cv.imwrite(save_folder+str(i)+'.jpg',mask0)

