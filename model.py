import torch.nn as nn
import torch

def _weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()

class Block(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Block,self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,5,1,2),
            nn.ReLU(inplace=True),
        )
        self.apply(_weights_init)
    def forward(self,x):
        x = self.block(x)
        return x

class DCNN(nn.Module):
    def __init__(self,block_num):
        super(DCNN,self).__init__()
        self.input = nn.Sequential(
            nn.Conv2d(1,64,5,1,2),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList()
        for _ in range(block_num-1):
            self.blocks.append(Block(64,64))
        self.out = nn.Sequential(
            nn.Conv2d(64,1,5,1,2),
        )
        self.apply(_weights_init)
    def forward(self,x):
        x = self.input(x)
        for block in self.blocks:
            x = block(x)
        x = self.out(x)
        return x

class RS(nn.Module):
    def __init__(self,block_num):
        super(RS,self).__init__()
        self.IRS = DCNN(16)
        self.ISMP = DCNN(6)
        self.input = nn.Sequential(
            nn.Conv2d(3,64,5,1,2),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList()
        for _ in range(block_num-1):
            self.blocks.append(Block(64,64))
        self.out = nn.Sequential(
            nn.Conv2d(64,1,5,1,2),
        )
        self.apply(_weights_init)

    def initModel(self,device):
        state_dict = torch.load('./model/IRS_best_psnr.pkl',map_location=device)
        self.IRS.load_state_dict(state_dict['model'],strict=False)
        print('IRS model load done, epoch:%d, best_loss:%f, best psnr:%f'
              % (state_dict['epoch'],state_dict['best_loss'],state_dict['best_psnr']))

    def forward(self,x):
        map1 = self.IRS(x)
        map2 = self.ISMP(map1)
        y = self.input(torch.cat((x,map1,map2),1))
        for block in self.blocks:
            y = block(y)
        out = self.out(y)
        return map2,out

