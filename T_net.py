import torch as t
import torch.nn as nn
import torch.nn.functional as F


class T_Neural_net(t.nn.Module):
    def __init__(self):
        super(T_Neural_net,self).__init__()
        self.fc1 = nn.Linear(28 * 28, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, 10)
        self.relu = nn.ReLU()
        

    
    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = F.dropout(x, p=0.8, training=self.training)
        x = self.fc2(x)
        x = self.relu(x)
        x = F.dropout(x, p=0.8, training=self.training)
        x = self.fc3(x)
        return x





