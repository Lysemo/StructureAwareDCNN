import torch
import torch.nn as nn
import torch.nn.functional as F

class MseLoss(nn.Module):
    def __init__(self):
        super(MseLoss,self).__init__()
        self.mseLoss = nn.MSELoss()
    def forward(self,pre,gt):
        loss = self.mseLoss(pre,gt)
        return loss

class JointMseLoss(nn.Module):
    def __init__(self):
        super(JointMseLoss,self).__init__()
        self.mseLoss = nn.MSELoss()
    def forward(self,pre1,pre2,gt,sobel_gt):
        loss1 = self.mseLoss(pre1,sobel_gt)
        loss2 = self.mseLoss(pre2,gt)
        loss = loss1 + loss2
        return loss