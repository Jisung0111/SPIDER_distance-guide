# contains opening preprocessed data and overall management.

import torch as th
import numpy as np
import argparse
import Code.model

parser = argparse.ArgumentParser();

parser.add_argument('--seed', defalut = 0, type = int);
parser.add_argument('--epochs', default = 50, type = int);
parser.add_argument('--batch_size', default = 100, type = int);
parser.add_argument('--lr', default = 0.001, type = float);
parser.add_argument('--input_size', default = '224_224', type = str); # 224 * 224 * 3 -> '224_224' # Original VGG-19 and Resnet get 224x224 input.
parser.add_argument('--batch_norm', default = 1, type = int); # indicates to use batch norm
parser.add_argument('--feature_dim', default = 128, type = int); # dimension of output of CNN.
parser.add_argument('--guide', default = 'distance', type = str); # 'distance' or 'none'
parser.add_argument('--neural_net', default = 'resnet', type = str); # 'vgg'(VGG-19) or 'resnet'(Resnet-50)

class Model:
    def __init__(self, )

def main():
    pass

    # Data Loading

    # model Loading

    # Learning Process (included in model.py)

if __name__ == '__main__':
    args = parser.parse_args();
    main();
