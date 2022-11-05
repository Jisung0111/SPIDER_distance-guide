# contains neural_networks (VGG-19, ResNet)

import torch as th
import torch.nn as nn
import numpy as np

VGG_19_layers = [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"];

class VGG_19(nn.Module):
    def __init__(self, input_size, feature_dim, batch_norm):
        super(VGG_19, self).__init__();
        if input_size == "224_224":
            self.layers = nn.ModuleList();

            in_ch = 3;
            for v in VGG_19_layers:
                if v == "M":
                    self.layers.append(nn.MaxPool2d((2, 2), (2, 2)));
                else:
                    self.layers.append(nn.Conv2d(in_ch, v, (3, 3), (1, 1), (1, 1)));
                    if batch_norm:
                        self.layers.append(nn.BatchNorm2d(v));
                    self.layers.append(nn.ReLU(True));
                in_ch = v;

            self.avg_pool = nn.AdaptiveAvgPool2d((7, 7));

            self.fc_layer = nn.ModuleList([
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(4096, feature_dim)
            ]);
    
    def forward(self, x):
        x = self.layers(x);
        x = self.avg_pool(x);
        x = self.fc_layer(x.view(-1));
        return x;

class Bottleneck(nn.Module):
    expansion = 4;
    def __init__ (self, in_ch, out_ch, stride, batch_norm):
        super(Bottleneck, self).__init__();
        self.layers = nn.ModuleList();
        
        self.layers.append(nn.Conv2d(in_ch, out_ch, 1));
        if batch_norm: self.layers.append(nn.BatchNorm2d(out_ch));
        self.layers.append(nn.ReLU(True));
        self.layers.append(nn.Conv2d(out_ch, out_ch, 3, stride, 1));
        if batch_norm: self.layers.append(nn.BatchNorm2d(out_ch));
        self.layers.append(nn.ReLU(True));
        self.layers.append(nn.Conv2d(out_ch, self.expansion * out_ch, 1));
        if batch_norm: self.layers.append(nn.BatchNorm2d(out_ch));

        self.shortcut = nn.ModuleList();
        if stride != 1 or in_ch != self.expansion * out_ch:
            self.shortcut.append(nn.Conv2d(out_ch, self.expansion * out_ch, 1, stride));
            if batch_norm: self.layers.append(nn.BatchNorm2d(out_ch));
        
        self.relu = nn.ReLU(True);
    
    def forward(self, x):
        return self.relu(self.layers(x) + self.shortcut(x));

class ResNet_50(nn.Module):
    def __init__(self, input_size, feature_dim, batch_norm):
        super(ResNet_50, self).__init__();
        if input_size == "224_224":
            self.layers = nn.ModuleList();
