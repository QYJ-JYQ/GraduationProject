import time
from pynq.overlays.base import BaseOverlay
from pynq.lib.video import *
import numpy as np
from pynq import Xlnk
import struct
from scipy.misc import imread
import cv2
from pynq import GPIO

ol=BaseOverlay("base.bit")
ol.ip_dict
ol.download()
conv=ol.Conv_0
pool=ol.Pool_0

print("Overlay download finish");

def readbinfile(filename,size):
    f = open(filename, "rb")
    z=[]
    for j in range(size):
        data = f.read(4)
        data_float = struct.unpack("f", data)[0]
        z.append(data_float)
    f.close()
    z = np.array(z)
    return z
def RunConv(conv,Kx,Ky,Sx,Sy,mode,relu_en,feature_in,W,bias,feature_out):
    conv.write(0x10,feature_in.shape[2]);
    conv.write(0x18,feature_in.shape[0]);
    conv.write(0x20,feature_in.shape[1]);
    conv.write(0x28,feature_out.shape[2]);
    conv.write(0x30,Kx);
    conv.write(0x38,Ky);
    conv.write(0x40,Sx);
    conv.write(0x48,Sy);
    conv.write(0x50,mode);
    conv.write(0x58,relu_en);
    conv.write(0x60,feature_in.physical_address);
    conv.write(0x68,W.physical_address);
    conv.write(0x70,bias.physical_address);
    conv.write(0x78,feature_out.physical_address);
    conv.write(0, (conv.read(0)&0x80)|0x01 );
    tp=conv.read(0)
    while not ((tp>>1)&0x1):
        tp=conv.read(0);
def RunPool(pool,Kx,Ky,mode,feature_in,feature_out):
    pool.write(0x10,feature_in.shape[2]);
    pool.write(0x18,feature_in.shape[0]);
    pool.write(0x20,feature_in.shape[1]);
    pool.write(0x28,Kx);
    pool.write(0x30,Ky);
    pool.write(0x38,mode);
    pool.write(0x40,feature_in.physical_address);
    pool.write(0x48,feature_out.physical_address);
    pool.write(0, (pool.read(0)&0x80)|0x01 );
    while not ((pool.read(0)>>1)&0x1):
        pass;
    
#Conv0
IN_WIDTH0=68
IN_HEIGHT0=68
IN_CH0=1

KERNEL_WIDTH0=5
KERNEL_HEIGHT0=5
X_STRIDE0=1
Y_STRIDE0=1

RELU_EN0=1
MODE0=0  #0:VALID, 1:SAME
if(MODE0):
    X_PADDING0=int((KERNEL_WIDTH1-1)/2)
    Y_PADDING0=int((KERNEL_HEIGHT1-1)/2)
else:
    X_PADDING0=0
    Y_PADDING0=0

OUT_CH0=1
OUT_WIDTH0=int((IN_WIDTH0+2*X_PADDING0-KERNEL_WIDTH0)/X_STRIDE0+1)
OUT_HEIGHT0=int((IN_HEIGHT0+2*Y_PADDING0-KERNEL_HEIGHT0)/Y_STRIDE0+1)


#Pool0
MODE00=2  #mode: 0:MEAN, 1:MIN, 2:MAX
IN_WIDTH00=OUT_WIDTH0
IN_HEIGHT00=OUT_HEIGHT0
IN_CH00=OUT_CH0

KERNEL_WIDTH00=2
KERNEL_HEIGHT00=2

OUT_CH00=IN_CH00
OUT_WIDTH00=int(IN_WIDTH00/KERNEL_WIDTH00)
OUT_HEIGHT00=int(IN_HEIGHT00/KERNEL_HEIGHT00)
    
#Conv1
IN_WIDTH1=OUT_WIDTH00
IN_HEIGHT1=OUT_HEIGHT00
IN_CH1=1

KERNEL_WIDTH1=5
KERNEL_HEIGHT1=5
X_STRIDE1=1
Y_STRIDE1=1

RELU_EN1=1
MODE1=0  #0:VALID, 1:SAME
if(MODE1):
    X_PADDING1=int((KERNEL_WIDTH1-1)/2)
    Y_PADDING1=int((KERNEL_HEIGHT1-1)/2)
else:
    X_PADDING1=0
    Y_PADDING1=0

OUT_CH1=1
OUT_WIDTH1=int((IN_WIDTH1+2*X_PADDING1-KERNEL_WIDTH1)/X_STRIDE1+1)
OUT_HEIGHT1=int((IN_HEIGHT1+2*Y_PADDING1-KERNEL_HEIGHT1)/Y_STRIDE1+1)


#Pool1
MODE11=2  #mode: 0:MEAN, 1:MIN, 2:MAX
IN_WIDTH11=OUT_WIDTH1
IN_HEIGHT11=OUT_HEIGHT1
IN_CH11=OUT_CH1

KERNEL_WIDTH11=2
KERNEL_HEIGHT11=2

OUT_CH11=IN_CH11
OUT_WIDTH11=int(IN_WIDTH11/KERNEL_WIDTH11)
OUT_HEIGHT11=int(IN_HEIGHT11/KERNEL_HEIGHT11)

#Conv2
IN_WIDTH2=OUT_WIDTH11
IN_HEIGHT2=OUT_HEIGHT11
IN_CH2=OUT_CH11

KERNEL_WIDTH2=5
KERNEL_HEIGHT2=5
X_STRIDE2=1
Y_STRIDE2=1

RELU_EN2=1
MODE2=0  #0:VALID, 1:SAME
if(MODE2):
    X_PADDING2=int((KERNEL_WIDTH2-1)/2)
    Y_PADDING2=int((KERNEL_HEIGHT2-1)/2)
else:
    X_PADDING2=0
    Y_PADDING2=0

OUT_CH2=2
OUT_WIDTH2=int((IN_WIDTH2+2*X_PADDING2-KERNEL_WIDTH2)/X_STRIDE2+1)
OUT_HEIGHT2=int((IN_HEIGHT2+2*Y_PADDING2-KERNEL_HEIGHT2)/Y_STRIDE2+1)

#Pool2
MODE21=2  #mode: 0:MEAN, 1:MIN, 2:MAX
IN_WIDTH21=OUT_WIDTH2
IN_HEIGHT21=OUT_HEIGHT2
IN_CH21=OUT_CH2

KERNEL_WIDTH21=2
KERNEL_HEIGHT21=2

OUT_CH21=IN_CH21
OUT_WIDTH21=int(IN_WIDTH21/KERNEL_WIDTH21)
OUT_HEIGHT21=int(IN_HEIGHT21/KERNEL_HEIGHT21) #5, 5, 16

#Fc1
IN_WIDTH3=OUT_WIDTH21
IN_HEIGHT3=OUT_HEIGHT21
IN_CH3=OUT_CH21

KERNEL_WIDTH3=5
KERNEL_HEIGHT3=5
X_STRIDE3=1
Y_STRIDE3=1

RELU_EN3=1
MODE3=0  #0:VALID, 1:SAME
if(MODE3):
    X_PADDING3=int((KERNEL_WIDTH3-1/2))
    Y_PADDING3=int((KERNEL_HEIGHT3-1)/2)
else:
    X_PADDING3=0
    Y_PADDING3=0

OUT_CH3=60
OUT_WIDTH3=int((IN_WIDTH3+2*X_PADDING3-KERNEL_WIDTH3)/X_STRIDE3+1)
OUT_HEIGHT3=int((IN_HEIGHT3+2*Y_PADDING3-KERNEL_HEIGHT3)/Y_STRIDE3+1)
#print(OUT_WIDTH3, OUT_HEIGHT3)
#Fc2
IN_WIDTH4=OUT_WIDTH3
IN_HEIGHT4=OUT_HEIGHT3
IN_CH4=OUT_CH3

KERNEL_WIDTH4=1
KERNEL_HEIGHT4=1
X_STRIDE4=1
Y_STRIDE4=1

RELU_EN4=1
MODE4=0  #0:VALID, 1:SAME
if(MODE4):
    X_PADDING4=int((KERNEL_WIDTH4-1/2))
    Y_PADDING4=int((KERNEL_HEIGHT4-1)/2)
else:
    X_PADDING4=0
    Y_PADDING4=0

OUT_CH4=10
OUT_WIDTH4=int((IN_WIDTH4+2*X_PADDING4-KERNEL_WIDTH4)/X_STRIDE4+1)
OUT_HEIGHT4=int((IN_HEIGHT4+2*Y_PADDING4-KERNEL_HEIGHT4)/Y_STRIDE4+1)


xlnk=Xlnk();

#input image
image=xlnk.cma_array(shape=(IN_HEIGHT0,IN_WIDTH0,IN_CH0),cacheable=0,dtype=np.float32)

#conv0
W_conv0=xlnk.cma_array(shape=(KERNEL_HEIGHT0,KERNEL_WIDTH0,IN_CH0,OUT_CH0),cacheable=0,dtype=np.float32)
b_conv0=xlnk.cma_array(shape=(OUT_CH0),cacheable=0,dtype=np.float32)
h_conv0=xlnk.cma_array(shape=(OUT_HEIGHT0,OUT_WIDTH0,OUT_CH0),cacheable=0,dtype=np.float32)
h_pool0=xlnk.cma_array(shape=(OUT_HEIGHT00,OUT_WIDTH00,OUT_CH00),cacheable=0,dtype=np.float32)

#conv1
W_conv1=xlnk.cma_array(shape=(KERNEL_HEIGHT1,KERNEL_WIDTH1,IN_CH1,OUT_CH1),cacheable=0,dtype=np.float32)
b_conv1=xlnk.cma_array(shape=(OUT_CH1),cacheable=0,dtype=np.float32)
h_conv1=xlnk.cma_array(shape=(OUT_HEIGHT1,OUT_WIDTH1,OUT_CH1),cacheable=0,dtype=np.float32)
h_pool1=xlnk.cma_array(shape=(OUT_HEIGHT11,OUT_WIDTH11,OUT_CH11),cacheable=0,dtype=np.float32)

#conv2
W_conv2=xlnk.cma_array(shape=(KERNEL_HEIGHT2,KERNEL_WIDTH2,IN_CH2,OUT_CH2),cacheable=0,dtype=np.float32)
b_conv2=xlnk.cma_array(shape=(OUT_CH2),cacheable=0,dtype=np.float32)
h_conv2=xlnk.cma_array(shape=(OUT_HEIGHT2,OUT_WIDTH2,OUT_CH2),cacheable=0,dtype=np.float32)
h_pool2=xlnk.cma_array(shape=(OUT_HEIGHT21,OUT_WIDTH21,OUT_CH21),cacheable=0,dtype=np.float32)

#fc1
W_fc1=xlnk.cma_array(shape=(KERNEL_HEIGHT3, KERNEL_WIDTH3, IN_CH3, OUT_CH3),cacheable=0,dtype=np.float32)
b_fc1=xlnk.cma_array(shape=(OUT_CH3),cacheable=0,dtype=np.float32)
h_fc1=xlnk.cma_array(shape=(OUT_HEIGHT3,OUT_WIDTH3,OUT_CH3),cacheable=0,dtype=np.float32)

#fc2
W_fc2=xlnk.cma_array(shape=(KERNEL_HEIGHT4, KERNEL_WIDTH4, IN_CH4, OUT_CH4),cacheable=0,dtype=np.float32)
b_fc2=xlnk.cma_array(shape=(OUT_CH4),cacheable=0,dtype=np.float32)
h_fc2=xlnk.cma_array(shape=(OUT_HEIGHT4,OUT_WIDTH4,OUT_CH4),cacheable=0,dtype=np.float32)



print("start initial");
#Initialize W, bias
w_conv0=readbinfile("./weight_v2/conv2d_weight.bin",KERNEL_HEIGHT0*KERNEL_WIDTH0*IN_CH0*OUT_CH0)
w_conv0=w_conv0.reshape((KERNEL_HEIGHT0,KERNEL_WIDTH0,IN_CH0,OUT_CH0))
for i in range(KERNEL_HEIGHT0):
    for j in range(KERNEL_WIDTH0):
        for k in range(IN_CH0):
            for l in range(OUT_CH0):
                W_conv0[i][j][k][l]=w_conv0[i][j][k][l]
print("finish w_conv0");

B_conv0=readbinfile("./weight_v2/conv2d_bias.bin",OUT_CH0)
for i in range(OUT_CH0):
    b_conv0[i]=B_conv0[i]
print("finish B_conv0");

w_conv1=readbinfile("./weight_v2/conv2d_1_weight.bin",KERNEL_HEIGHT1*KERNEL_WIDTH1*IN_CH1*OUT_CH1)
w_conv1=w_conv1.reshape((KERNEL_HEIGHT1,KERNEL_WIDTH1,IN_CH1,OUT_CH1))
for i in range(KERNEL_HEIGHT1):
    for j in range(KERNEL_WIDTH1):
        for k in range(IN_CH1):
            for l in range(OUT_CH1):
                W_conv1[i][j][k][l]=w_conv1[i][j][k][l]
print("finish w_conv1");

B_conv1=readbinfile("./weight_v2/conv2d_1_bias.bin",OUT_CH1)
for i in range(OUT_CH1):
    b_conv1[i]=B_conv1[i]
print("finish B_conv1");

w_conv2=readbinfile("./weight_v2/conv2d_2_weight.bin",KERNEL_HEIGHT2*KERNEL_WIDTH2*IN_CH2*OUT_CH2)
w_conv2=w_conv2.reshape((KERNEL_HEIGHT2,KERNEL_WIDTH2,IN_CH2,OUT_CH2))
for i in range(KERNEL_HEIGHT2):
    for j in range(KERNEL_WIDTH2):
        for k in range(IN_CH2):
            for l in range(OUT_CH2):
                W_conv2[i][j][k][l]=w_conv2[i][j][k][l]
print("finish w_conv2");

B_conv2=readbinfile("./weight_v2/conv2d_2_bias.bin",OUT_CH2)
for i in range(OUT_CH2):
    b_conv2[i]=B_conv2[i]
print("finish B_conv2");


w_fc1=readbinfile("./weight_v2/conv2d_3_weight.bin",KERNEL_HEIGHT3*KERNEL_WIDTH3*IN_CH3*OUT_CH3)
w_fc1=w_fc1.reshape((KERNEL_HEIGHT3,KERNEL_WIDTH3,IN_CH3,OUT_CH3))
for i in range(KERNEL_HEIGHT3):
    for j in range(KERNEL_WIDTH3):
        for k in range(IN_CH3):
            for l in range(OUT_CH3):
                W_fc1[i][j][k][l]=w_fc1[i][j][k][l]
print("finish w_fc1");

B_fc1=readbinfile("./weight_v2/conv2d_3_bias.bin",OUT_CH3)
for i in range(OUT_CH3):
    b_fc1[i]=B_fc1[i]
print("finish B_fc1");

w_fc2=readbinfile("./weight_v2/dense_weight.bin",KERNEL_HEIGHT4*KERNEL_WIDTH4*IN_CH4*OUT_CH4)
w_fc2=w_fc2.reshape((KERNEL_HEIGHT4,KERNEL_WIDTH4,IN_CH4,OUT_CH4))
for i in range(KERNEL_HEIGHT4):
    for j in range(KERNEL_WIDTH4):
        for k in range(IN_CH4):
            for l in range(OUT_CH4):
                W_fc2[i][j][k][l]=w_fc2[i][j][k][l]
print("finish w_fc2");

B_fc2=readbinfile("./weight_v2/dense_bias.bin",OUT_CH4)
for i in range(OUT_CH4):
    b_fc2[i]=B_fc2[i]
print("finish B_fc2");


# monitor configuration: 640*480 @ 60Hz
Mode = VideoMode(640,480,24)
hdmi_out = ol.video.hdmi_out
hdmi_out.configure(Mode,PIXEL_BGR)
hdmi_out.start()

# camera (input) configuration
frame_in_w = 640
frame_in_h = 480

# initialize camera from OpenCV
import cv2

videoIn = cv2.VideoCapture(0)
videoIn.set(cv2.CAP_PROP_FRAME_WIDTH, frame_in_w);
videoIn.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_in_h);
print("Capture device is open: " + str(videoIn.isOpened()))
kernel = np.ones((3,3),np.uint8) 

import time
    
print("开始手势识别")  

save_path='./capture_photo_v2/'
i_h=0
j_h=-1
k_h=0
flag=0
while(1):
    i_h+=1
    yuprocess_start=time.time()
   
    ret, frame_vga = videoIn.read()
    frame_vga_in = frame_vga[:,:,:]
    img_ycrcb = cv2.cvtColor(frame_vga_in,cv2.COLOR_BGR2YCR_CB)
    (y, cr, cb) = cv2.split(img_ycrcb)
    cr1 = cv2.GaussianBlur(cr, (5, 5), 0)
    _, skin = cv2.threshold(cr1,133,173,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img = cv2.resize(skin,(68,68)) 
    
    #img = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)
    image1 = img/255.0
    image1 = image1.reshape((IN_HEIGHT0,IN_WIDTH0,IN_CH0))
    np.copyto(image,image1)
    yuprocess_end=time.time()
    
    hdmi_start=time.time()
    
    if (ret): 
        skin=np.reshape(skin,(480,640,1))
        outframe = hdmi_out.newframe()
        outframe[0:480,0:640,:] = skin[0:480,0:640,:]
        hdmi_out.writeframe(outframe)
    else:
        raise RuntimeError("Failed to read from camera.")
         
    hdmi_end=time.time()
    #预处理结束

    
    wangluo_start=time.time()
    #conv0
    RunConv(conv,KERNEL_WIDTH0,KERNEL_HEIGHT0,X_STRIDE0,Y_STRIDE0,MODE0,RELU_EN0,image,W_conv0,b_conv0,h_conv0)
    #pool0
    RunPool(pool, KERNEL_WIDTH00, KERNEL_HEIGHT00, MODE00, h_conv0, h_pool0)
    #conv1
    RunConv(conv,KERNEL_WIDTH1,KERNEL_HEIGHT1,X_STRIDE1,Y_STRIDE1,MODE1,RELU_EN1,h_pool0,W_conv1,b_conv1,h_conv1)
    #pool1
    RunPool(pool, KERNEL_WIDTH11, KERNEL_HEIGHT11, MODE11, h_conv1, h_pool1)
    #conv2
    RunConv(conv,KERNEL_WIDTH2,KERNEL_HEIGHT2,X_STRIDE2,Y_STRIDE2,MODE2,RELU_EN2,h_pool1,W_conv2,b_conv2,h_conv2)
    # pool2
    RunPool(pool, KERNEL_WIDTH21, KERNEL_HEIGHT21, MODE21, h_conv2, h_pool2)
    # fc1
    RunConv(conv,KERNEL_WIDTH3,KERNEL_HEIGHT3,X_STRIDE3,Y_STRIDE3,MODE3,RELU_EN3,h_pool2,W_fc1,b_fc1,h_fc1)
    # fc2
    RunConv(conv,KERNEL_WIDTH4,KERNEL_HEIGHT4,X_STRIDE4,Y_STRIDE4,MODE4,RELU_EN4,h_fc1,W_fc2,b_fc2,h_fc2)
    
    result=0
    max=h_fc2[0][0][0]
    for j in range(1,OUT_CH4):
        if(h_fc2[0][0][j]>max):
            max=h_fc2[0][0][j]
            result=j 
    end=time.time()
    print( "手势识别结果：",result,"预处理耗时：%5.4f" %(yuprocess_end-yuprocess_start-0.002),"神经网络耗时：%5.4f" %(end-wangluo_start-0.002),"HDMI输出耗时：%5.4f" %(hdmi_end-hdmi_start-0.002))
    if(ol.buttons[3].read()==1):
        videoIn.release()
        hdmi_out.stop()
        del hdmi_out
        print("手势识别结束")
        break
    if(ol.buttons[1].read()==1):
        j_h+=1
        k_h=0
        save_path=save_path+'%s'%(str(j_h))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    if(ol.buttons[2].read()==1):
        flag=~flag
    if(flag):
        if(k_h==500):
            print('该手势已够500张')
            continue
        cv2.imwrite(save_path+'/%s.jpg'%(str(i_h+20000)), skin)
        k_h+=1
        print("保存成功",'手势：',j_h,'第',k_h,'张')