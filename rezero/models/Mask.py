import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SandGlassBlock(nn.Module):

    def __init__(self, in_c):
        super().__init__()
        self.linear1 = nn.Linear(in_features=in_c,
                                 out_features=in_c * 2,
                                 bias=False)
        self.bn1 = nn.BatchNorm1d(in_c * 2)
        self.linear2 = nn.Linear(in_features=in_c * 2,
                                 out_features=in_c,
                                 bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        # output = self.linear1(x)
        # output = self.bn1(output)
        # output = self.relu(output)
        # output = self.linear2(output)
        output = self.sigmoid(x)
        # output = self.tanh(output)
        # print(x)
        
        # output = self.sigmoid(x)
        # print(output[0])
        # exit()
        return output
    
class FeatureMask(nn.Module):

    def __init__(self):

        super().__init__()
        self.in_c = 64

        self.mask_module = SandGlassBlock(self.in_c)
        # self.zerotensor = torch.tensor(0, dtype = torch.float32).cuda()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight) 
    
    def forward(self, input):
        # zerotensor = torch.tensor(0, dtype = torch.float32).cuda()
        batch_size, ch, h, w = input.shape
        input_new = input.clone().detach()
        input_pool = input_new.mean(dim=1).reshape(batch_size, -1)
        
        out = self.mask_module(input_pool)
        # print(out.dtype)
        # out = torch.where(out < 0.5, zerotensor, out)
        out = out.reshape(batch_size, h, w).unsqueeze(dim = 1)
        # print(out.shape) batch * hw
        # print(input.shape) batch * ch * h * w
        # exit()
        feat_mask = out * input
        bg_mask = (1 - out) * input
        
        return feat_mask, bg_mask
        