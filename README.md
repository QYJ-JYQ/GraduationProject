# GraduationProject
固定摄像头，重新采集了数据集，并进行了参数训练

data_32*32_500采用500张32*32的图片进行训练，上板测试结果一般

加大了训练集的图片数,得到data_32_32_1000参数文件夹，上板测试结果反而更差了

分析：最初的数据集手势比较保守，后来的数据集中手势角度刁钻，加上图片尺寸压缩，特征丢失较多

下一步将会增大图片尺寸，进一步训练。

2022.3.14