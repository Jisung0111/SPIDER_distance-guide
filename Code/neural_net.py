# contains neural_networks (VGG, ResNet)

import torch as th
import torch.nn as nn
import numpy as np

class VGG(nn.Module):
    def __init__(self, input_size, feature_dim, batch_norm, VGG_layers):
        super(VGG, self).__init__();
        if input_size == "224_224":
            self.layers = nn.ModuleList();
            in_ch = 3;
            for n, ch in zip(VGG_layers, [64, 128, 256, 512, 512]):
                for _ in range(n):
                    self.layers.append(nn.Conv2d(in_ch, ch, 3, 1, 1));
                    if batch_norm: self.layers.append(nn.BatchNorm2d(ch));
                    self.layers.append(nn.ReLU(True));
                in_ch = ch;
                self.layers.append(nn.MaxPool2d(2, 2));
            self.layers.append(nn.AdaptiveAvgPool2d((7, 7)));
            self.layers = nn.Sequential(*self.layers);

            self.fc_layer = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(4096, feature_dim)
            );
        else:
            raise ValueError("Wrong input size");
    
    def forward(self, x):
        x = self.layers(x);
        x = self.fc_layer(x.view((x.shape[0], -1)));
        return x;

def VGG_11(input_size, feature_dim, batch_norm):
    return VGG(input_size, feature_dim, batch_norm, [1, 1, 2, 2, 2]);

def VGG_13(input_size, feature_dim, batch_norm):
    return VGG(input_size, feature_dim, batch_norm, [2, 2, 2, 2, 2]);

def VGG_16(input_size, feature_dim, batch_norm):
    return VGG(input_size, feature_dim, batch_norm, [2, 2, 3, 3, 3]);

def VGG_19(input_size, feature_dim, batch_norm):
    return VGG(input_size, feature_dim, batch_norm, [2, 2, 4, 4, 4]);



class ResNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, batch_norm, BasicBlock, expansion):
        super(ResNetBlock, self).__init__();

        self.layers = nn.ModuleList();
        if BasicBlock:
            self.layers.append(nn.Conv2d(in_ch, out_ch, 3, stride, 1));
            if batch_norm: self.layers.append(nn.BatchNorm2d(out_ch));
            self.layers.append(nn.ReLU(True));
            self.layers.append(nn.Conv2d(out_ch, out_ch, 3, 1, 1));
        else:
            self.layers.append(nn.Conv2d(in_ch, out_ch, 1));
            if batch_norm: self.layers.append(nn.BatchNorm2d(out_ch));
            self.layers.append(nn.ReLU(True));
            self.layers.append(nn.Conv2d(out_ch, out_ch, 3, stride, 1));
            if batch_norm: self.layers.append(nn.BatchNorm2d(out_ch));
            self.layers.append(nn.ReLU(True));
            self.layers.append(nn.Conv2d(out_ch, expansion * out_ch, 1));
        if batch_norm: self.layers.append(nn.BatchNorm2d(expansion * out_ch));
        self.layers = nn.Sequential(*self.layers);

        self.shortcut = nn.ModuleList();
        if stride != 1 or in_ch != expansion * out_ch:
            self.shortcut.append(nn.Conv2d(in_ch, expansion * out_ch, 1, stride));
            if batch_norm: self.shortcut.append(nn.BatchNorm2d(expansion * out_ch));
        self.shortcut = nn.Sequential(*self.shortcut);
        
        self.relu = nn.ReLU(True);
    
    def forward(self, x):
        return self.relu(self.layers(x) + self.shortcut(x));

class ResNet(nn.Module):
    def __init__(self, input_size, feature_dim, batch_norm, ResNet_layers, BasicBlock):
        super(ResNet, self).__init__();
        if input_size == "224_224":
            self.layers = nn.ModuleList();
            expansion = 1 if BasicBlock else 4;

            in_ch = 64;
            self.layers.append(nn.Conv2d(3, in_ch, 7, 2, 3));
            if batch_norm: self.layers.append(nn.BatchNorm2d(in_ch));
            self.layers.append(nn.ReLU(True));
            self.layers.append(nn.MaxPool2d(3, 2, 1));
            for n, ch, st in zip(ResNet_layers, [64, 128, 256, 512], [1, 2, 2, 2]):
                self.layers.append(ResNetBlock(in_ch, ch, st, batch_norm, BasicBlock, expansion));
                in_ch = ch * expansion;
                for _ in range(n - 1):
                    self.layers.append(ResNetBlock(in_ch, ch, 1, batch_norm, BasicBlock, expansion));
            self.layers.append(nn.AdaptiveAvgPool2d((1, 1)));
            self.layers = nn.Sequential(*self.layers);
            
            self.fc_layer = nn.Linear(512 * expansion, feature_dim);
        else:
            raise ValueError("Wrong input size");
    
    def forward(self, x):
        x = self.layers(x);
        x = self.fc_layer(x.view((x.shape[0], -1)));
        return x;

def ResNet_18(input_size, feature_dim, batch_norm):
    return ResNet(input_size, feature_dim, batch_norm, [2, 2, 2, 2], True);

def ResNet_34(input_size, feature_dim, batch_norm):
    return ResNet(input_size, feature_dim, batch_norm, [3, 4, 6, 3], True);

def ResNet_50(input_size, feature_dim, batch_norm):
    return ResNet(input_size, feature_dim, batch_norm, [3, 4, 6, 3], False);

def ResNet_101(input_size, feature_dim, batch_norm):
    return ResNet(input_size, feature_dim, batch_norm, [3, 4, 23, 3], False);

def ResNet_152(input_size, feature_dim, batch_norm):
    return ResNet(input_size, feature_dim, batch_norm, [3, 8, 36, 3], False);
