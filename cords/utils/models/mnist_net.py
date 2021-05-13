import torch
import torch.nn as nn
import torch.nn.functional as F


class MnistNet(nn.Module):
    def __init__(self, kernel1=3, feat_size1 = 32, kernel2=3, feat_size2=64, drop1=0.25, drop2=0.5, fc_deep=1, fc_width=128):
        super(MnistNet, self).__init__()
        self.embDim = 128
        
        self.conv1 = nn.Conv2d(1, feat_size1, kernel1, 1)
        self.conv2 = nn.Conv2d(32, feat_size2, kernel2, 1)
        self.dropout1 = nn.Dropout2d(drop1)
        self.dropout2 = nn.Dropout2d(drop2)
        self.fc1 = []
        for i in range(fc_deep):
            self.fc1.append(nn.Linear(9216, fc_width))
        self.fc1 = nn.Sequential(*self.fc1)
        self.fc2 = nn.Linear(128, 10)


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
            out = self.fc1(out)
            out = F.relu(out)
            e = self.dropout2(out)
        out = self.fc2(e)
        if last:
            return out, e
        else:
            return out


    def get_embedding_dim(self):
        return self.embDim
