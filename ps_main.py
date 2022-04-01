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


def conv_s(kernel,input1):
    val = sum(sum(kernel*input1))
    return val

def Convolution(FeatureMap,Kernal,Bias):
    Height_FM=FeatureMap.shape[0]
    Width_FM=FeatureMap.shape[1]

    Height_kernal=Kernal.shape[0]
    Width_Kernal=Kernal.shape[1]
    InChannelNum=Kernal.shape[2]
    OutChannelNum=Kernal.shape[3]

    OutFM_Height=Height_FM-Height_kernal+1
    OutFM_Width=Width_FM-Width_Kernal+1

    MiddleData0=np.zeros((OutFM_Height,OutFM_Width))
    MiddleData1=np.zeros((OutFM_Height,OutFM_Width))
    Result=np.zeros((OutFM_Height,OutFM_Width,OutChannelNum))
    for L in range(OutChannelNum):
        for K in range(InChannelNum):
            for J in range(OutFM_Height):
                for I in range(OutFM_Width):
                    MiddleData0[I,J]=conv_s(Kernal[:,:,K,L],FeatureMap[I:I+Height_kernal,J:J+Width_Kernal,K])
            MiddleData1+=MiddleData0
        Result[:,:,L]=MiddleData1[:,:]+Bias[L]
        MiddleData0[:,:]=0
        MiddleData1[:,:]=0
    Result[np.where(Result<0)]=0.0  #Relu
    return Result

def MaxPooling(FeatureMap):
    '''
        2*2最大池化，步长2
    '''
    Height_FM=FeatureMap.shape[0]
    Width_FM=FeatureMap.shape[1]
    Channel_FM=FeatureMap.shape[2]

    Height_Out_FM=int(Height_FM/2)
    Width_Out_FM=int(Width_FM/2)
    
    Result=np.zeros((Height_Out_FM,Width_Out_FM,Channel_FM))

    for L in range(Channel_FM):
        for K in range(Height_Out_FM):
            for J in range(Width_Out_FM):
                Result[J,K,L] = FeatureMap[2*J:(2*J+2),2*K:(2*K+2),L].max()
    
    return Result

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

#数组定义


print("start initial");
#Initialize W, bias
w_conv0=readbinfile("./weight_v2/conv2d_weight.bin",KERNEL_HEIGHT0*KERNEL_WIDTH0*IN_CH0*OUT_CH0)
w_conv0=w_conv0.reshape((KERNEL_HEIGHT0,KERNEL_WIDTH0,IN_CH0,OUT_CH0))
print("finish w_conv0");

B_conv0=readbinfile("./weight_v2/conv2d_bias.bin",OUT_CH0)
print("finish B_conv0");

w_conv1=readbinfile("./weight_v2/conv2d_1_weight.bin",KERNEL_HEIGHT1*KERNEL_WIDTH1*IN_CH1*OUT_CH1)
w_conv1=w_conv1.reshape((KERNEL_HEIGHT1,KERNEL_WIDTH1,IN_CH1,OUT_CH1))
print("finish w_conv1");

B_conv1=readbinfile("./weight_v2/conv2d_1_bias.bin",OUT_CH1)
print("finish B_conv1");

w_conv2=readbinfile("./weight_v2/conv2d_2_weight.bin",KERNEL_HEIGHT2*KERNEL_WIDTH2*IN_CH2*OUT_CH2)
w_conv2=w_conv2.reshape((KERNEL_HEIGHT2,KERNEL_WIDTH2,IN_CH2,OUT_CH2))
print("finish w_conv2");

B_conv2=readbinfile("./weight_v2/conv2d_2_bias.bin",OUT_CH2)
print("finish B_conv2");


w_fc1=readbinfile("./weight_v2/conv2d_3_weight.bin",KERNEL_HEIGHT3*KERNEL_WIDTH3*IN_CH3*OUT_CH3)
w_fc1=w_fc1.reshape((KERNEL_HEIGHT3,KERNEL_WIDTH3,IN_CH3,OUT_CH3))
print("finish w_fc1");

B_fc1=readbinfile("./weight_v2/conv2d_3_bias.bin",OUT_CH3)
print("finish B_fc1");

w_fc2=readbinfile("./weight_v2/dense_weight.bin",KERNEL_HEIGHT4*KERNEL_WIDTH4*IN_CH4*OUT_CH4)
w_fc2=w_fc2.reshape((KERNEL_HEIGHT4,KERNEL_WIDTH4,IN_CH4,OUT_CH4))
print("finish w_fc2");

B_fc2=readbinfile("./weight_v2/dense_bias.bin",OUT_CH4)
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

save_path_0='./capture_photo_v3/'
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
#     _, skin = cv2.threshold(cr,133,173,cv2.THRESH_BINARY)
    img = cv2.resize(skin,(68,68)) 
    
    #img = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)
    image1 = img/255.0
    image1 = image1.reshape((IN_HEIGHT0,IN_WIDTH0,IN_CH0))
    # np.copyto(image,image1)
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
    conv_1=Convolution(image1,w_conv0,B_conv0)
    pool_1=MaxPooling(conv_1)
    conv_2=Convolution(pool_1,w_conv1,B_conv1)
    pool_2=MaxPooling(conv_2)
    conv_3=Convolution(pool_2,w_conv2,B_conv2)
    pool_3=MaxPooling(conv_3)
    conv_4=Convolution(pool_3,w_fc1,B_fc1)
    fc_1=Convolution(conv_4,w_fc2,B_fc2)

    result=0
    max=fc_1[0][0][0]
    for j in range(1,OUT_CH4):
        if(fc_1[0][0][j]>max):
            max=fc_1[0][0][j]
            result=j 
    end=time.time()
    print( "手势识别结果：",result,"预处理耗时：%5.4f" %(yuprocess_end-yuprocess_start-0.002),"神经网络耗时：%5.4f" %(end-wangluo_start-0.002),"HDMI输出耗时：%5.4f" %(hdmi_end-hdmi_start-0.002))
    if(ol.buttons[3].read()==1):
        videoIn.release()
        hdmi_out.stop()
        del hdmi_out
        print("手势识别结束")
        break


    # 采集数据集
    if(ol.buttons[1].read()==1):
        j_h+=1
        k_h=0
        save_path=save_path_0+'%s'%(str(j_h))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    if(ol.buttons[2].read()==1):
        flag=~flag
    if(flag):
        if(k_h==500):
            print('该手势已够500张')
            continue
        cv2.imwrite(save_path+'/%s.jpg'%(str(i_h+20000)), frame_vga)
        k_h+=1
        print("保存成功",'手势：',j_h,'第',k_h,'张')