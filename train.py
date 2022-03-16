import tensorflow as tf
import numpy as np
import glob
import os
from skimage import io, transform, color
import time
from PIL import Image
import cv2 as cv
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model

np.set_printoptions(threshold=np.inf)

train_dataset_path='C:/Users/28155/Desktop/github/capture_photo_all/'
width=32
hight=32

def read_img(path):
    '''
    输入:数据集路径
    输出:图片,标签
    '''
    global cnt
    folder_name=[x for x in os.listdir(path) if os.path.isdir(path + x)]
    #0,1,10,11,2,3,...,9
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    imgs=[]
    labels=[]
    for index, folder in enumerate(cate):
        cnt=0
        for img_addr in glob.glob(folder+'/*.jpg'):
            print('reading the images:%s' %img_addr)
            img=cv.imread(img_addr,0)
            img = cv.resize(img,(32,32))
            img = np.array(img) 
            bool_30=img>=127
            img[bool_30]=255
            bool_30_=img<127
            img[bool_30_]=0 
            img=img/255. 
            imgs.append(img)
            labels.append(folder_name[index])
            cnt+=1
            if(cnt==1100):
                break
    imgs=np.array(imgs)
    labels=np.array(labels)
    labels = labels.astype(np.int64)
    return imgs, labels


#打乱顺序
x,y_=read_img(train_dataset_path)


np.random.seed(116)  # 使用相同的seed，保证输入特征和标签一一对应
np.random.shuffle(x)
np.random.seed(116)
np.random.shuffle(y_)

x=np.reshape(x,(1100*10,32,32,1))
# 分为训练集和测试集
ratio=0.8
num_flag=int(ratio*cnt*10)
x_train=x[:num_flag]
x_test=x[num_flag:]
y_train=y_[:num_flag]
y_test=y_[num_flag:]



x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)


class LeNet5(Model):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.c1 = Conv2D(filters=1, kernel_size=(5, 5),
                         activation='relu')
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2)

        self.c2 = Conv2D(filters=2, kernel_size=(5, 5),
                         activation='relu')
        self.p2 = MaxPool2D(pool_size=(2, 2), strides=2)

        self.c3 = Conv2D(filters=60, kernel_size=(5, 5),
                         activation='relu')

        self.flat = Flatten()

        self.f3 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.p1(x)

        x = self.c2(x)
        x = self.p2(x)
        
        x = self.c3(x)
        x = self.flat(x)
        # x = self.c3(x)
        # x = self.c4(x)
        # x = self.c3(x)
        # x = self.c4(x)
        y = self.f3(x)
        return y


model = LeNet5()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = "./checkpoint/LeNet5.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

history = model.fit(x_train, y_train, batch_size=32, epochs=200, validation_data=(x_test, y_test), validation_freq=1
                    ,callbacks=[cp_callback])
model.summary()

model.save('saved_model')

# print(model.trainable_variables)
file = open('./weights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

# 显示训练集和验证集的acc和loss曲线
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()




            
