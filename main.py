from config import opt
from Dataset import Data
import numpy as np
import torch as t
import torchvision as tv
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from T_net import T_Neural_net 
from S_net import S_Neural_net
import torch.nn.functional as F

# Tnet = t.load('Net')



def distillation(y, labels, teacher_scores, T, alpha):
    return nn.KLDivLoss()(F.log_softmax(y/T), F.softmax(teacher_scores/T)) * (T*T * 2.0 * alpha) + F.cross_entropy(y, labels) * (1. - alpha)

def StudentTrain():
    Snet = S_Neural_net()
    Snet.train().cuda()
    Tnet = t.load('./TNet')
    Tnet.eval()
    train_data = Data(opt.data_path)
    train_dataloader = DataLoader(train_data,batch_size=opt.batch_size,shuffle=True)
    criterion = nn.CrossEntropyLoss()
    loss_fn = nn.KLDivLoss()

    optimizer = t.optim.SGD(Snet.parameters(),lr=opt.lr_1)
    for epoch in range(opt.max_epoch):
        print('current epoch:%s'%epoch)
        for i,(img,label) in enumerate(train_dataloader):
            img, label = Variable(img), Variable(label)
     
            optimizer.zero_grad()
            img = img.float().cuda()
            label = label.long().cuda()

            T_probe = nn.functional.softmax(Tnet(img)/opt.T)
            # TeacherLoss = criterion(T_probe,label)
            S_probe_1 = nn.functional.softmax(Snet(img)/opt.T)

            # loss_1 = (opt.T)*(opt.T)*loss_fn(S_probe_1,T_probe)

            S_probe_2 = nn.functional.softmax(Snet(img))
            loss_2 = criterion(S_probe_2,label)
            # StudentLoss = (1-opt.lamda)*loss_1 + opt.lamda*loss_2
            StudentLoss = distillation(Snet(img),label,Tnet(img),T=20,alpha=0.7)
            StudentLoss.backward()
            optimizer.step()
            if i%10==0:
                print('student_loss:%5.5f'%StudentLoss.data[0])
    t.save(Snet,'student_net')

def TeacherTrain():
    net = T_Neural_net()
    net.cuda()
    train_data = Data(opt.data_path)
    train_dataloader = DataLoader(train_data,batch_size=opt.batch_size,shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = t.optim.SGD(net.parameters(),lr=opt.lr)
    for epoch in range(opt.max_epoch):
        print('current epoch:%s'%epoch)
        for i,(img,label) in enumerate(train_dataloader):
            optimizer.zero_grad()
            img = img.float().cuda()
            label = label.long().cuda()
            output = net(img)
            loss = criterion(output,label)
            loss.backward()
            optimizer.step()
            # print('%5.5f'%loss.data[0])
            if i%20==0:
                print('loss:%5.5f'%loss.data[0])
    t.save(net,'TNet')
def TeacherTest():
    correct = 0
    net = t.load('./TNet')
    net.cuda()
    test_data = Data(opt.data_path,mode='t10k')
    test_dataloader = DataLoader(test_data,batch_size=opt.batch_size,shuffle=True)
    total = len(test_data)
    for img,label in test_dataloader:
        img = img.float().cuda()
        out = net(img)
        a,predict = t.max(out.data,1)
        label = label.long().cuda()
        correct +=(predict==label).sum()

    acc = (100*correct/total).float().item()
    print('correct:%s'%correct.item())
    print('Accuracy=%2.2f'%acc)

def StudentTest():
    correct = 0
    net = t.load('./student_net')
    net.cuda()
    test_data = Data(opt.data_path,mode='t10k')
    test_dataloader = DataLoader(test_data,batch_size=opt.batch_size,shuffle=True)
    total = len(test_data)
    for img,label in test_dataloader:
        img = img.float().cuda()
        out = net(img)
        a,predict = t.max(out.data,1)
        label = label.long().cuda()
        correct +=(predict==label).sum()

    acc = (100*correct/total).float().item()
    print('student_correct:%s'%correct.item())
    print('student_Accuracy=%2.2f'%acc)

TeacherTrain()
TeacherTest()

StudentTrain()
StudentTest()