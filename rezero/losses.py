import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import normal


def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)   #目标类概率
    loss = (1 - p.detach()) ** gamma * input_values
    return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)


class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=False, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 /np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)                             #one-hot
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))  #取得对应位置的m   self.m_list
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)                                       #x的index位置换成x_m
        
        return F.cross_entropy(self.s*output, target, weight=self.weight)  #weight=self.weight
    

class GCLLoss(nn.Module):
    
    def __init__(self, cls_num_list, m=0.5, weight=None, s=30, train_cls=False, noise_mul = 1., gamma=0.):
        super(GCLLoss, self).__init__()
        cls_list = torch.cuda.FloatTensor(cls_num_list)
        m_list = torch.log(cls_list)
        m_list = m_list.max()-m_list
        self.m_list = m_list
        assert s > 0
        self.m = m
        self.s = s
        self.weight = weight
        self.simpler = normal.Normal(0, 1/3)
        self.train_cls = train_cls
        self.noise_mul = noise_mul
        self.gamma = gamma
           
                                         
    def forward(self, cosine, target):
        
        # print(cosine.shape, target.shape)
        
        index = torch.zeros_like(cosine, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
             
        noise = self.simpler.sample(cosine.shape).clamp(-1, 1).to(cosine.device) #self.scale(torch.randn(cosine.shape).to(cosine.device))  
        # print(noise)
        #cosine = cosine - self.noise_mul * noise/self.m_list.max() *self.m_list   
        cosine = cosine - self.noise_mul * noise.abs()/self.m_list.max() *self.m_list         
        output = torch.where(index, cosine-self.m, cosine)                    
        if self.train_cls:
            return focal_loss(F.cross_entropy(self.s*output, target, reduction='none', weight=self.weight), self.gamma)
        else:    
            return F.cross_entropy(self.s*output, target, weight=self.weight)     
 
class UniformLoss(nn.Module):
    
    def __init__(self, cls_num_list, num_class = 10, s = 30):
        super(UniformLoss, self).__init__()
        self.uniform = 1 / num_class
        self.s = s
        self.bsceloss = BSCELoss(cls_num_list).cuda()
        
    def forward(self, logit): #logit batch * class num
        # logit = F.softmax(logit * self.s, dim = 1)
        
        logit = F.softmax(logit, dim = 1)
        target = torch.ones_like(logit) * self.uniform
        # print('UFloss',target)
        # exit()
        # return self.bsceloss(logit, target)        
        return F.cross_entropy(logit, target)      
        
class MaskLoss(nn.Module):
    
    def __init__(self, target=0.5):
        super(MaskLoss, self).__init__()
        self.target = target
        
    def forward(self, mask):
        mask = mask.mean(dim = 1)
        loss = torch.pow(mask - self.target, 2)
        return loss.mean()
        
class BSCELoss(nn.Module):
    def __init__(self, cls_num_list, tau=1, weight=None):
        super(BSCELoss, self).__init__()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1)
        self.weight = weight
        # self.sampler = normal.Normal(0, 1/3)
        
    def forward(self, x, target):
        # noise = self.sampler.sample(x.shape).clamp(-1, 1).to(x.device)
        # noise = noise.abs() * self.m_list
        x_m = x + self.m_list #- noise
        return F.cross_entropy(x_m, target, weight=self.weight)
    
class BCELoss(nn.Module):
    def __init__(self, cls_num_list, tau=1, weight=None):
        super(BCELoss, self).__init__()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1)
        self.weight = weight
        # self.sampler = normal.Normal(0, 1/3)
        
    def forward(self, x, target):
        # noise = self.sampler.sample(x.shape).clamp(-1, 1).to(x.device)
        # noise = noise.abs() * self.m_list
        # print(target.shape)
        # exit()
        gt = target.view(-1, 1)
        batch_size, num_cls = x.shape
        m_list = self.m_list.expand(batch_size, num_cls)
        label = torch.arange(num_cls).expand(batch_size, num_cls).cuda()
        x_m = torch.where(label == gt, x + m_list, x)
        # x_m = x + self.m_list #- noise
        return F.cross_entropy(x_m, target, weight=self.weight)
    
class SharpLoss(nn.Module):
    def __init__(self, eps = 1e-5):
        super(SharpLoss, self).__init__()
        self.eps = eps
        
    def forward(self, logits):
        logits = F.softmax(logits, dim = 1)
        ent = - (logits * (logits + self.eps).log()).sum(dim = 1)
        mean = ent.mean()
        
        return mean
    
class DivLoss(nn.Module):
    def __init__(self, eps = 1e-5):
        super(DivLoss, self).__init__()
        self.eps = eps
        
    def forward(self, logits):
        logits = F.softmax(logits, dim = 1)
        mean = logits.mean(dim=0)
        ent = - (mean * (mean + self.eps).log()).sum()
        
        return ent
    
class SparseLoss(nn.Module):
    def __init__(self, eps = 1e-15):
        super(SparseLoss, self).__init__()
        self.eps = eps
        
    def forward(self, features, labels, cls_num_list):
        
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        m_list = torch.log(cls_num_list)
        m_list = m_list-m_list.min()
        m_list = m_list / m_list.max()
        
        idx = torch.index_select(m_list, 0, labels).view(-1, 1)
        
        features_norm = features / (torch.max(features, dim = 1, keepdim = True).values + self.eps)
        
        if features_norm.sum() > 0 :
            shrinked_value = torch.sum(torch.exp(features_norm), dim = 1, keepdim = True)
            
            summed_value = torch.sum(features_norm, dim = 1, keepdim = True)
            
            outputs = - shrinked_value / (summed_value + self.eps)
            
        return outputs.mean()