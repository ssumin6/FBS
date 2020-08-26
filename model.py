import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import global_avgpool2d, winner_take_all


class FBSConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, fbs=False, sparsity_ratio=1.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.fbs = fbs
        self.sparsity_ratio = sparsity_ratio

        if fbs:
            # set self.bn weight not trainable
            self.bn.weight.requires_grad = False
            self.gl = nn.Linear(in_channels, out_channels, bias=True)
            # SET GL layer's bias as 1 and weight as he initialization
            nn.init.kaiming_uniform_(self.gl.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(self.gl.bias, 1.0)

    def forward(self, x):
        if self.fbs:
            x = self.fbs_forward(x)
            return x
        else:
            x = self.original_forward(x)
            return x

    def original_forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x

    def fbs_forward(self, x):
        scale = global_avgpool2d(x)
        scale = F.relu(self.gl(scale))
        scale = winner_take_all(scale, self.sparsity_ratio) 
        x = self.conv(x)
        x = self.bn(x)
        x = torch.mul(scale.unsqueeze(2).unsqueeze(3), x)
        x = F.relu(x)
        return x, torch.norm(scale, p=1)

class CifarNet(nn.Module):
    def __init__(self, fbs=False, sparsity_ratio=1.0):
        super().__init__()
        self.layer0 = FBSConv2d(3, 64, 3, stride=1, padding=0, fbs=fbs, sparsity_ratio=sparsity_ratio)
        self.layer1 = FBSConv2d(64, 64, 3, stride=1, padding=1, fbs=fbs, sparsity_ratio=sparsity_ratio)
        self.layer2 = FBSConv2d(64, 128, 3, stride=2, padding=1, fbs=fbs, sparsity_ratio=sparsity_ratio)
        self.layer3 = FBSConv2d(128, 128, 3, stride=1, padding=1, fbs=fbs, sparsity_ratio=sparsity_ratio)
        self.layer4 = FBSConv2d(128, 128, 3, stride=1, padding=1, fbs=fbs, sparsity_ratio=sparsity_ratio)
        self.layer5 = FBSConv2d(128, 192, 3, stride=2, padding=1, fbs=fbs, sparsity_ratio=sparsity_ratio)
        self.layer6 = FBSConv2d(192, 192, 3, stride=1, padding=1, fbs=fbs, sparsity_ratio=sparsity_ratio)
        self.layer7 = FBSConv2d(192, 192, 3, stride=1, padding=1, fbs=fbs, sparsity_ratio=sparsity_ratio)

        self.fbs = fbs
        self.pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(192, 10)

    def forward(self, x):
        if self.fbs:
            x, g = self.fbs_forward(x)
            return x, g
        else:
            x = self.original_forward(x)
            return x
        
    def original_forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        
        x = self.fc(x)

        return x

    def fbs_forward(self, x):
        g = 0 
        layers = [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.layer6, self.layer7]
        for layer in layers:
            x, tmp = layer(x)
            g += tmp

        x = self.pool(x)
        x = torch.flatten(x, 1)
        
        x = self.fc(x)

        return x, g
