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
import random

__all__ = ['ResNet_s', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Parameter):
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
                 classifier = True, use_norm= False, use_noise = False, drop_rate = 0.75):
        super(ResNet_s, self).__init__()
        self.in_planes = 16
        self.classifier = classifier
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.pecent = 1/3
        self.drop_rate = drop_rate
        self.obj_vector = nn.Parameter(torch.rand([100, 64]), requires_grad = True)
        
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

    def forward(self, x, gt = None, flag = None, feat = None, cls_num_list = None):
        
        
                
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        if flag:
            # interval = 10
            # if epoch % interval == 0:
            #     self.pecent = 3.0 / 10 + (epoch / interval) * 2.0 / 10
            
            cls_num_list = torch.cuda.FloatTensor(cls_num_list)
            m_list = torch.log(cls_num_list)
            m_list = m_list-m_list.min()
            m_list = m_list / m_list.max()
            
            idx = torch.index_select(m_list, 0, gt)

            self.eval()
            x_new = x.clone().detach()
            x_new = Variable(x_new.data, requires_grad=True)  #batch channel H W
            
            x_new_view = F.avg_pool2d(x_new, x_new.size()[3])                       
            x_new_view = x_new_view.view(x_new_view.size(0), -1)  # batch channel
            
            output = self.linear(x_new_view)
            
            class_num = output.shape[1]
            index = gt
            num_rois = x_new.shape[0]  # batch
            num_channel = x_new.shape[1] # channel
            H = x_new.shape[2]           # H
            HW = x_new.shape[2] * x_new.shape[3]  # HxW
            
            one_hot = torch.zeros((1), dtype=torch.float32).cuda() 
            one_hot = Variable(one_hot, requires_grad=False)
            
            sp_i = torch.ones([2, num_rois]).long()  # 2 X Batch
            sp_i[0, :] = torch.arange(num_rois) # 0~batchsize
            sp_i[1, :] = index                  # batch target
            sp_v = torch.ones([num_rois])       # [1, 1, 1, ...] batch size
            
            one_hot_sparse = torch.sparse.FloatTensor(sp_i, sp_v, torch.Size([num_rois, class_num])).to_dense().cuda()
            one_hot_sparse = Variable(one_hot_sparse, requires_grad=False) # (batch, cls_num) batch내 target클래스 값 1 else 0
            one_hot = torch.sum(output * one_hot_sparse) #(64, 10) * (64, 10) {batch, num_classes} 
            
            # return x, x_new, one_hot
            
            # grads_val = torch.autograd.grad(one_hot, x_new)
            # grads_val = torch.stack(list(grads_val), dim = 0).squeeze()#.clone.detach()

            self.zero_grad()
            one_hot.backward()
            
            grads_val = x_new.grad.clone().detach()
            # print("zerograd:", grads_val)
            
            grads_val = grads_val.view(num_rois, num_channel, -1)
            grad_channel_mean = torch.mean(grads_val, dim=2) #batch channel hw
            channel_mean = grad_channel_mean # batch channel 1
            
            grad_channel_mean = grad_channel_mean.view(num_rois, num_channel, 1, 1)
            spatial_mean = torch.sum(x_new * grad_channel_mean, 1) # (b c h w) * (b c 1 1) ---(mean)---> b h w
            spatial_mean = spatial_mean.view(num_rois, HW) # b hw
            
            # spatial_mean = torch.mean(grads_val, dim = 1)
            # spatial_mean = spatial_mean.view(num_rois, HW)
            
            self.zero_grad()

            choose_one = random.randint(0, 9)
            if choose_one <= 4:
                # ---------------------------- spatial -----------------------
                spatial_drop_num = torch.ceil(HW * 1 / 4.0 * idx).long().view(-1, 1)
                th18_mask_value = torch.sort(spatial_mean, dim=1, descending=True)[0] #batch x 1 (threshold)
                th18_mask_value = torch.gather(th18_mask_value, 1, spatial_drop_num)
                th18_mask_value = th18_mask_value.view(num_rois, 1).expand(num_rois, HW) # 49 -> 64
                mask_all_cuda = torch.where(spatial_mean > th18_mask_value, torch.zeros(spatial_mean.shape).cuda(),
                                            torch.ones(spatial_mean.shape).cuda()) # b hw
                mask_all = mask_all_cuda.reshape(num_rois, H, H).view(num_rois, 1, H, H)
                
                # att_mask = torch.sigmoid(spatial_mean).reshape(num_rois, H, H).view(num_rois, 1, H, H)
                
                
            else:
                # -------------------------- channel ----------------------------
                vector_thresh_percent = torch.ceil(num_channel * 1 / 4 * idx).long().view(-1, 1)
                vector_thresh_value = torch.sort(channel_mean, dim=1, descending=True)[0]
                vector_thresh_value = torch.gather(vector_thresh_value, 1, vector_thresh_percent)
                vector_thresh_value = vector_thresh_value.view(num_rois, 1).expand(num_rois, num_channel)
                vector = torch.where(channel_mean > vector_thresh_value,
                                     torch.zeros(channel_mean.shape).cuda(),
                                     torch.ones(channel_mean.shape).cuda())
                mask_all = vector.view(num_rois, num_channel, 1, 1)
                
                # att_mask = torch.sigmoid(channel_mean).view(num_rois, num_channel, 1, 1)

            # ----------------------------------- batch ----------------------------------------
            
            rand_tensor = torch.rand([], dtype=torch.float32) + self.drop_rate
            binary = rand_tensor.floor()
            
            self.train()
            mask_all = Variable(mask_all, requires_grad=True)
            
            if binary:
                x = x * mask_all
            # else:
            #     x = x * att_mask

            # cls_prob_before = F.softmax(output, dim=1)
            
            # x_new_view_after = x_new * mask_all
            # x_new_view_after = F.avg_pool2d(x_new_view_after, x_new_view_after.size()[3])
            # x_new_view_after = x_new_view_after.view(x_new_view_after.size(0), -1)
            # x_new_view_after = self.linear(x_new_view_after)
            # cls_prob_after = F.softmax(x_new_view_after, dim=1)

            # sp_i = torch.ones([2, num_rois]).long() # 2 X Batch
            # sp_i[0, :] = torch.arange(num_rois)     # 0~batchsize
            # sp_i[1, :] = index                      # batch target
            # sp_v = torch.ones([num_rois])           # [1, 1, 1, ...] batch size
            
            # one_hot_sparse = torch.sparse.FloatTensor(sp_i, sp_v, torch.Size([num_rois, class_num])).to_dense().cuda()
            
            # before_vector = torch.sum(one_hot_sparse * cls_prob_before, dim=1) #batch x 1
            # after_vector = torch.sum(one_hot_sparse * cls_prob_after, dim=1) #batch x 1
            
            # change_vector = before_vector - after_vector - 0.0001
            # change_vector = torch.where(change_vector > 0, change_vector, torch.zeros(change_vector.shape).cuda())
            
            # th_fg_value = torch.sort(change_vector, dim=0, descending=True)[0][int(round(float(num_rois) * self.pecent))]
            # drop_index_fg = change_vector.gt(th_fg_value).long() #thresh 초과 값 1
            # ignore_index_fg = 1 - drop_index_fg #thresh 이하 값 1
            # not_01_ignore_index_fg = ignore_index_fg.nonzero()[:, 0] # thresh 이하 값을 가지는 행요소 뽑기
            # mask_all[not_01_ignore_index_fg.long(), :] = 1 # mask_all -> ch (b, c, 1, 1), sp (b, 1, h, w)

            
        
        # return out
        feat = F.avg_pool2d(x, x.size()[3])
        #feat=F.dropout2d(feat, p=self.dropout_rate)  #0831 加了dropput,
        feat = feat.view(feat.size(0), -1)

        score = self.linear(feat)
        return score
    
    # def grad_mask(self, x, x_new):
        
    #     # class_num = output.shape[1]
    #     # index = gt
        
    #     num_rois = x_new.shape[0]  # batch
    #     num_channel = x_new.shape[1] # channel
    #     H = x_new.shape[2]           # H
    #     HW = x_new.shape[2] * x_new.shape[3]  # HxW
        
    #     grads_val = x_new.grad.clone().detach()
    #     grad_channel_mean = torch.mean(grads_val.view(num_rois, num_channel, -1), dim=2) #batch channel hw
    #     channel_mean = grad_channel_mean # batch channel 1
        
    #     # grad_channel_mean = grad_channel_mean.view(num_rois, num_channel, 1, 1)
    #     # spatial_mean = torch.sum(x_new * grad_channel_mean, 1) # (b c h w) * (b c 1 1) ---(mean)---> b h w
    #     # spatial_mean = spatial_mean.view(num_rois, HW) # b hw
        
    #     spatial_mean = torch.mean(grads_val.view(num_rois, num_channel, -1), dim = 1)
    #     spatial_mean = spatial_mean.view(num_rois, HW)
        
    #     # self.zero_grad()
    #     choose_one = random.randint(0, 9)
    #     if choose_one <= 4:
    #         # ---------------------------- spatial -----------------------
    #         spatial_drop_num = math.ceil(HW * 1 / 3.0)
    #         th18_mask_value = torch.sort(spatial_mean, dim=1, descending=True)[0][:, spatial_drop_num] #batch x 1 (threshold)
    #         th18_mask_value = th18_mask_value.view(num_rois, 1).expand(num_rois, HW) # 49 -> 64
    #         mask_all_cuda = torch.where(spatial_mean > th18_mask_value, torch.zeros(spatial_mean.shape).cuda(),
    #                                     torch.ones(spatial_mean.shape).cuda()) # b hw
    #         mask_all = mask_all_cuda.reshape(num_rois, H, H).view(num_rois, 1, H, H)
    #     else:
    #         # -------------------------- channel ----------------------------
    #         vector_thresh_percent = math.ceil(num_channel * 1 / 3.2)
    #         vector_thresh_value = torch.sort(channel_mean, dim=1, descending=True)[0][:, vector_thresh_percent]
    #         vector_thresh_value = vector_thresh_value.view(num_rois, 1).expand(num_rois, num_channel)
    #         vector = torch.where(channel_mean > vector_thresh_value,
    #                                 torch.zeros(channel_mean.shape).cuda(),
    #                                 torch.ones(channel_mean.shape).cuda())
    #         mask_all = vector.view(num_rois, num_channel, 1, 1)

    #     # ----------------------------------- batch ----------------------------------------
        
    #     rand_tensor = torch.rand([], dtype=torch.float32) + self.drop_rate
    #     binary = rand_tensor.floor()
        
    #     # self.train()
    #     mask_all = Variable(mask_all, requires_grad=True)
        
    #     if binary:
    #         x = x * mask_all
            
    #     feat = F.avg_pool2d(x, x.size()[3])
    #     #feat=F.dropout2d(feat, p=self.dropout_rate)  #0831 加了dropput,
    #     feat = feat.view(feat.size(0), -1)

    #     score = self.linear(feat)
    #     return score

            

def resnet20():
    return ResNet_s(BasicBlock, [3, 3, 3])


def resnet32(num_classes=10, classifier=True, use_norm = False, use_noise=False, drop_rate = 0.75):
    return ResNet_s(BasicBlock, [5, 5, 5], num_classes=num_classes, 
                    classifier = classifier, use_norm= use_norm, use_noise=use_noise, drop_rate = drop_rate)


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
