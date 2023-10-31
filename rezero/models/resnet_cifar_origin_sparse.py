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

__all__ = ['ResNet_s', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.s = 30

    def forward(self, x):
        # out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        out = x.mm(F.normalize(self.weight, dim=0))
        # return self.s * out
        return out
    
class BatchLinear(nn.Module):
    
    def __init__(self, in_features, out_features):
        super(BatchLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        # self.ln = nn.LayerNorm(in_features)
        
    def forward(self, x):
        
        size = self.weight.size()
        
        total_mean = self.weight.mean()
        total_var = torch.sqrt(self.weight.var(dim = 1) + 1e-5).mean()
        
        cls_mean = self.weight.mean(dim = 1, keepdim = True)
        cls_var = torch.sqrt(self.weight.var(dim = 1, keepdim = True) + 1e-5)
        
        normal_weight = (self.weight - cls_mean.expand(size)) / cls_var.expand(size)
        adain_weight = total_var.expand(size) * normal_weight + total_mean.expand(size)
        
        out = x.mm(adain_weight)
        return out
    
class AdaINLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(AdaINLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        # self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.out_features = out_features
        # nn.init.constant_(self.bias.data, 0)
        # self.mean = nn.Parameter(torch.Tensor(1))
        # self.var = nn.Parameter(torch.Tensor(1))

    def forward(self, x):
        batch_size, _ = x.shape
        feat_size = x.size()
        size = self.weight.size()
        
        cls_mean = self.weight.mean(dim = 0, keepdim = True)
        cls_var = torch.sqrt(self.weight.var(dim = 0, keepdim = True) + 1e-5)
        
        total_mean = self.weight.mean()
        # total_var = torch.sqrt(torch.cuda.FloatTensor([0.1264]))
        total_var = torch.sqrt(self.weight.var(dim  = 0, keepdim = True) + 1e-5).mean()
        
        ##################################################
        # if iter == 0 and epoch == 0:
        
        #     feat_total_mean = x.mean().detach()
        #     feat_total_var = torch.sqrt(x.var(dim = 1) + 1e-5).mean().detach()

        #     self.mean = nn.Parameter(feat_total_mean)
        #     self.var = nn.Parameter(feat_total_var)
        
        # feat_ins_mean = x.mean(dim = 1, keepdim = True)
        # feat_ins_var = torch.sqrt(x.var(dim = 1, keepdim = True) + 1e-5)
        
        # normal_feat = (x - feat_ins_mean.expand(feat_size)) / feat_ins_var.expand(feat_size)
        # adain_feat = total_var.expand(feat_size) * normal_feat + total_mean.expand(feat_size)
        
        ###################################################
        
        normal_weight = (self.weight - cls_mean.expand(size)) / cls_var.expand(size)
        adain_weight = total_var.expand(size) * normal_weight + total_mean.expand(size)
        
        cosine = x.mm(adain_weight)#+ self.bias.expand(batch_size, self.out_features)

        return cosine    
    
class UniformLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(UniformLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.out_features = out_features

    def forward(self, x):
        batch_size, _ = x.shape
        print(self.weight)
        
        cls_mean = self.weight.mean(dim = 0, keepdim = True)
        cls_var = torch.sqrt(self.weight.var(dim = 0, keepdim = True) + 1e-8)
        
        total_mean = self.weight.mean()
        print(total_mean)
        exit()
        # total_var = torch.sqrt(self.weight.var(dim = 0) + 1e-8).mean().clone().detach()
        
        cls_min, _ = self.weight.min(dim = 0, keepdim = True) # 1 x classes
        cls_max, _ = self.weight.max(dim = 0, keepdim = True) # 1 x classes
        uni_var = torch.sqrt((cls_max - cls_min)**2 / 12 + 1e-8)
        # print(uni_var.shape)
        # print('cla_var_mean:', total_var)
        # print('total_var:', torch.sqrt(self.weight.var() + 1e-8))
        total_var = uni_var
        
        adain_weight = total_var * (self.weight - cls_mean) / cls_var + total_mean
        # print(adain_weight.shape)
        
        # adain_weight = (self.weight - cls_mean) / cls_var
        
        cosine = x.mm(adain_weight) + self.bias.expand(batch_size, self.out_features)
        
        return cosine

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
        # self.ln = nn.LayerNorm([64, 8, 8])
        # self.bn2 = nn.BatchNorm2d(64)
        # self.IN = nn.InstanceNorm2d(64, track_running_stats = True)
        self.GN = nn.GroupNorm(2, 64)
    
        if self.classifier:
            if use_norm == 'norm':
                self.linear = NormedLinear(64, num_classes)
            elif use_norm == 'uniform':
                self.linear = UniformLinear(64, num_classes)
            elif use_norm == 'adain':
                self.linear = AdaINLinear(64, num_classes)
            elif use_norm == 'batch':
                self.linear = BatchLinear(64, num_classes)
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

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # return out
        # out = self.ln(out)
        # out = self.bn2(out)
        # out = self.IN(out)
        out = self.GN(out)
        feat = F.avg_pool2d(out, out.size()[3])
        feat = feat.view(feat.size(0), -1)
        # feat = self.ln(feat)
        # feat = self.bn2(feat)
        # feat = 0.5 * bn_feat + 0.5 * ln_feat

        score = self.linear(feat)
        
        return score
            

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
