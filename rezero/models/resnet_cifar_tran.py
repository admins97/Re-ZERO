'''
Properly implemented ResNet for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter
from torch.distributions import normal
from torch.autograd import Variable
from .Mask import FeatureMask

__all__ = ['ResNet_s', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        cosine = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        #out = x.mm(self.weight)
        return cosine * 30

class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)      
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_s(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10, 
                 classifier = True, use_norm= False, use_noise = False):
        super(ResNet_s, self).__init__()
        self.in_planes = 16
        self.classifier = classifier
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.drop_rate = 0.5 
        self.bn2 = nn.BatchNorm1d(num_classes)
        
        if self.classifier:
            if use_norm:
                self.linear = NormedLinear(64, num_classes)
            else:
                self.linear = nn.Linear(64, num_classes)
                
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, gt = None, flag = None, flag2 =None, cls_num_list = None):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # return out
        # out shape = 256 * 64 * 8 * 8 
        # print(out.shape)
        # return out  # for featuremap
        # feat, bg = self.feat_mask(out)
        # return feat, mask # for attention feature map
        if flag:
            
            cls_num_list = torch.cuda.FloatTensor(cls_num_list)
            m_list = torch.log(cls_num_list)
            m_list = m_list.max()-m_list
            m_list = m_list / m_list.max()
            
            idx = torch.index_select(m_list, 0, gt)
            
            cls_weight = self.linear.weight.T.clone().detach() # num_cls, channel
            
            batch_weight = torch.index_select(cls_weight, 0, gt) # batch, channel
            batch_weight = F.softmax(batch_weight, dim = 1).unsqueeze(dim = -1).unsqueeze(dim = -1)
                
            batch_size, ch, h, w = out.shape
            out_new = out.clone().detach()
            spatial_mean = out_new.mean(dim=1).reshape(batch_size, -1) 
            
            clsfr_spatial_mean = (out_new * batch_weight).sum(dim = 1).reshape(batch_size, -1)
            
            sp_drop_num = math.ceil(h * w * 0.5)
            sp_th_value = torch.sort(spatial_mean, dim = 1, descending=True)[0][:, sp_drop_num]
            sp_th_value = sp_th_value.view(batch_size, 1).expand(batch_size, h * w)
            sp_mask = torch.where(spatial_mean > sp_th_value, torch.zeros(spatial_mean.shape).cuda(), clsfr_spatial_mean)
            
            diff_spatial_mean = sp_mask - spatial_mean
        
            drop_num = torch.ceil(h * w * 0.25 * idx).long().view(-1, 1)
            th_mask_value = torch.sort(diff_spatial_mean, dim=1, descending = True)[0]
            th_mask_value = torch.gather(th_mask_value, 1, drop_num)
            th_mask_value = th_mask_value.view(batch_size, 1).expand(batch_size, h * w)
            
            spatial_back_mask = torch.where(diff_spatial_mean > th_mask_value, torch.zeros(diff_spatial_mean.shape).cuda(),
                                            torch.ones(diff_spatial_mean.shape).cuda())
            spatial_back_mask = spatial_back_mask.reshape(batch_size, h, w).unsqueeze(dim = 1)
            spatial_back_mask = Variable(spatial_back_mask, requires_grad = True)
            
            
            
            rand_tensor = torch.rand([], dtype=torch.float32) + self.drop_rate
            binary = rand_tensor.floor()
            
            if binary:
                out = out * spatial_back_mask
            
        feat = F.avg_pool2d(out, out.size()[3])
        #feat=F.dropout2d(feat, p=self.dropout_rate)  #0831 加了dropput,
        feat = feat.view(feat.size(0), -1)
        
        # bg = F.avg_pool2d(bg, bg.size()[3])
        # bg = bg.view(bg.size(0), -1)

        feat_score = self.linear(feat)
        if flag2:
            norm_score = self.bn2(feat_score)
            return feat_score, norm_score
        # bg_score = self.linear(bg)

        return feat_score
            

def resnet20():
    return ResNet_s(BasicBlock, [3, 3, 3])


def resnet32(num_classes=10, classifier=True, use_norm = False, use_noise=False):
    return ResNet_s(BasicBlock, [5, 5, 5], num_classes=num_classes, 
                    classifier = classifier, use_norm= use_norm, use_noise=use_noise )


def resnet44():
    return ResNet_s(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet_s(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet_s(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet_s(BasicBlock, [200, 200, 200])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()
