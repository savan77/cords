import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class MnistNet(nn.Module):
    def __init__(self, kernel1=3, feat_size1 = 32, kernel2=3, feat_size2=64, drop1=0.25, drop2=0.5, fc_deep=1, fc_width=128, batch_size=20):
        super(MnistNet, self).__init__()
        self.embDim = 128
        print("bs", batch_size)
        self.conv1 = nn.Conv2d(1, feat_size1, kernel1, 1)
        self.conv2 = nn.Conv2d(feat_size1, feat_size2, kernel2, 1)
        self.dropout1 = nn.Dropout2d(drop1)
        self.dropout2 = nn.Dropout2d(drop2)
        size = self.get_flat_fts((1,28,28), nn.Sequential(self.conv1, self.conv2))
        size = (size // 4)
        print("sze****",size)
        self.fc1 = []
        for i in range(fc_deep):
            self.fc1.append(nn.Linear(size, fc_width))
            size = fc_width
        self.fc1 = nn.Sequential(*self.fc1)
        print(self.fc1)
        self.fc2 = nn.Linear(fc_width, 10)

    def get_flat_fts(self, in_size, fts):
        f = fts(Variable(torch.ones(1,*in_size)))
        return int(np.prod(f.size()[1:]))

    def forward(self, x, last=False, freeze=False):
        if freeze:
            with torch.no_grad():
                out = self.conv1(x)
                out = F.relu(out)
                out = self.conv2(out)
                out = F.relu(out)
                out = F.max_pool2d(out, 2)
                out = self.dropout1(out)
                out = torch.flatten(out, 1)
                print("out freeze", out.size())
                out = self.fc1(out)
                out = F.relu(out)
                e = self.dropout2(out) 
        else:
            out = self.conv1(x)
            out = F.relu(out)
            out = self.conv2(out)
            out = F.relu(out)
            out = F.max_pool2d(out, 2)
            out = self.dropout1(out)
            out = torch.flatten(out, 1)
            print("out dize", out.size())
            print("**", self.fc1)
            out = self.fc1(out)
            out = F.relu(out)
            e = self.dropout2(out)
            print("fc2 size", e.size())
        out = self.fc2(e)
        if last:
            return out, e
        else:
            return out


    def get_embedding_dim(self):
        return self.embDim
