# contains learning process

import torch as th
import numpy as np

class Model:
    def __init__(self, epoch, lr, batch_size, input_size, guide, neural_net):
        self.epoch = epoch;
        self.lr = lr;
        self.batch_size = batch_size;
        self.guide = guide;
        if neural_net == 'vgg':
            from Code.neural_net import VGG_19
            self.neural_net = VGG_19(input_size);
        elif neural_net == 'resnet':
            from Code.neural_net import ResNet_50
            self.neural_net = ResNet_50(input_size);