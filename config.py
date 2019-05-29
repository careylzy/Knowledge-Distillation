#coding:utf8

class DefaultConfig(object):
    num_classes = 10;
    max_epoch = 15;
    batch_size = 256;
    lr =  3*1e-3;
    # lr = 1e-4;

    lr_1 = 1e-4
    data_path = '.\MNIST_DATA'
    T = 20   #temperature
    lamda = 0.2 #loss weight
    momentum = 0.5



opt = DefaultConfig()



