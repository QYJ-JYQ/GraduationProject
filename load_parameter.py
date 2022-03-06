import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model

save_path='D:/Microelectronics/GraduationProject/MainProject/cnn/data_v2/'


def Record_Tensor(array, name):
    print("Recording tensor " + name + " ...")
    f = open(save_path + name + '.dat', 'w+')
    # print ("The range: ["+str(np.min(array))+":"+str(np.max(array))+"]")
    if (np.size(np.shape(array)) == 1):
        Record_Array1D(array, name, f)
    else:
        if (np.size(np.shape(array)) == 2):
            Record_Array2D(array, name, f)
        else:
            if (np.size(np.shape(array)) == 3):
                Record_Array3D(array, name, f)
            else:
                Record_Array4D(array, name, f)
    f.close();


def Record_Array1D(array, name, f):
    for i in range(np.shape(array)[0]):
        f.write(str(array[i]) + "\n");


def Record_Array2D(array, name, f):
    for i in range(np.shape(array)[0]):
        for j in range(np.shape(array)[1]):
            f.write(str(array[i][j]) + "\n");


def Record_Array3D(array, name, f):
    for i in range(np.shape(array)[0]):
        for j in range(np.shape(array)[1]):
            for k in range(np.shape(array)[2]):
                f.write(str(array[i][j][k]) + "\n");


def Record_Array4D(array, name, f):
    for i in range(np.shape(array)[0]):
        for j in range(np.shape(array)[1]):
            for k in range(np.shape(array)[2]):
                for l in range(np.shape(array)[3]):
                    f.write(str(array[i][j][k][l]) + "\n");

np.set_printoptions(threshold=np.inf)

model=tf.keras.models.load_model('saved_model_v2')
model.summary()
layers=model.layers

for layer in layers:
    weight,bias=np.empty(2)
    # print(layer.name)
    if((layer.name).find('pool')==-1):
        if((layer.name).find('flat')==-1):
            weight, bias=model.get_layer(layer.name).get_weights()
            Record_Tensor(weight,layer.name+'_weight')
            Record_Tensor(bias,layer.name+'_bias')
        else:
            continue
    else:
        continue

# f = open(save_path + 'weight_0' + '.dat', 'w+')
# Record_Array4D(weight_0,'weight_0',f)

