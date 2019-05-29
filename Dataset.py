#coding:UTF-8
import os
import struct
from torch.utils import data
import numpy as np
from config import opt


class Data(data.Dataset):
    def __init__(self,path,mode='train'):
        labels_path = os.path.join(path,'%s-labels.idx1-ubyte'%mode)
        images_path = os.path.join(path,'%s-images.idx3-ubyte'%mode)
        with open(labels_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II',lbpath.read(8))
            self.labels = labels = np.fromfile(lbpath,dtype=np.uint8)
        with open(images_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
            self.imgs = images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 784)
    

    def __getitem__(self,index):
        img = self.imgs[index]
        label = self.labels[index]
        return img,label
    
    def __len__(self):
        return len(self.imgs)

