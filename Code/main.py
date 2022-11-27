# contains opening preprocessed data and overall management.

import torch as th
import numpy as np
import argparse
from model import Model
import time
import json
import pickle
import utils

parser = argparse.ArgumentParser();

parser.add_argument('--seed', default = 0, type = int);
parser.add_argument('--epochs', default = 50, type = int);
parser.add_argument('--batch_size', default = 100, type = int); # batch size for training set. Should divide 16000.
parser.add_argument('--vbatch_size', default = 500, type = int); # batch size for valid, test set. Should divide 2000.
parser.add_argument('--data_per_figr', default = 10, type = int);
parser.add_argument('--lr', default = 0.001, type = float);
parser.add_argument('--lr_scheduler', default = 'None', type = str); # one of [None, ReduceLROnPlateau, CosineAnnealingLR]
parser.add_argument('--input_size', default = '224_224', type = str); # 224 * 224 * 3 -> '224_224' # Original VGG-19 and Resnet get 224x224 input.
parser.add_argument('--batch_norm', default = 1, type = int); # indicates to use batch norm
parser.add_argument('--loss_setting', default = 0, type = int); # if loss setting is 0, general loss. or 1, only changes Y.
parser.add_argument('--feature_dim', default = 64, type = int); # dimension of output of CNN.
parser.add_argument('--guide', default = 'Distance', type = str); # 'Distance' or 'None'
parser.add_argument('--tau', default = 10.0, type = float); # used to determine necessity of distance guidance. still not sure how much value is appropriate.
parser.add_argument('--reg', default = 0.5, type = float); # weight of reg_loss. still not sure how much value is appropriate.
parser.add_argument('--Q', default = 10.0, type = float); # used for calculating hyper parameter alpha, beta, gamma.
parser.add_argument('--neural_net', default = 'ResNet-50', type = str); # one of {VGG-11, VGG-13, VGG-16, VGG-19, ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152}
parser.add_argument('--device', default = 'cuda:0', type = str);

data_split = {'train': 0, 'valid': 1, 'test': 2};

def main(args):
    start_T = time.time();
    utils.seed_init(args.seed);

    labels = [np.load("../Data/label_{}.npy".format(split)) for split in data_split];

    model = Model(
        args.lr,
        args.lr_scheduler,
        args.batch_size,
        args.data_per_figr,
        args.input_size,
        args.feature_dim,
        args.batch_norm,
        args.guide,
        args.neural_net,
        th.device(args.device),
        args.tau,
        args.reg,
        args.loss_setting,
        args.Q
    );

    # Make a result saving directory
    result_path = utils.make_result_dir("../Results");
    with open(result_path + "hparam.json", 'w') as f: json.dump(vars(args), f, indent = 4);

    history = {
        "epoch": [],
        "best_epoch": 0,
        "loss": [],
        "train_avgdist": [],
        "valid_avgdist": [],
        "valid_acc": [],
        "test_avgdist": 0,
        "test_acc": 0
    }

    # Learning Process
    try:
        epoch = 0;
        while epoch < args.epochs:
            epoch, loss, train_avgdist, train_stddist, train_distribution, valid_avgdist, valid_acc = \
                model.learn(epoch, labels[0], labels[1], args.vbatch_size);
            history["epoch"].append(epoch);
            history["loss"].append(loss);
            history["train_avgdist"].append(train_avgdist);
            history["valid_avgdist"].append(valid_avgdist);
            history["valid_acc"].append(valid_acc);
            with open(result_path + "Training_Log.txt", 'a') as f:
                f.write("Epoch {} ({})\tLoss: {:.4f}\tTrain Dist: {:.4f} +/- {:.4f}\tValid AvgDist: {:.4f}\tValid Acc: {:.4f}\n".format(
                    epoch, utils.hms(int(time.time() - start_T)), loss, train_avgdist, train_stddist, valid_avgdist, valid_acc
                ));
            if max(history["valid_acc"]) == valid_acc:
                history["best_epoch"] = epoch;
                model.save_model(result_path + "model.pth");
            utils.plot_train_distribution(result_path, epoch, train_distribution, train_avgdist, args.tau);
    except Exception as e:
        with open(result_path + "Training_Log.txt", 'a') as f:
            f.write("Accidently Training Stopped\nError Message: {}".format(repr(e)));

    # Finishing
    model.neural_net.load_state_dict(th.load(result_path + "model.pth", map_location = args.device));
    history["test_avgdist"], history["test_acc"] = utils.get_test_val(model.neural_net, labels[2], args.device, args.vbatch_size);
    with open(result_path + "Training_Log.txt", 'a') as f:
        f.write("\nTraining Done ({})\nModel with Best Accuracy on Validation set (Epoch: {})\nTest AvgDist: {:.4f}\tTest Acc: {:.4f}\n".format(
            utils.hms(int(time.time() - start_T)), history["best_epoch"], history["test_avgdist"], history["test_acc"]
        ));

    utils.plot_graph(history, result_path);

    del model;
    if args.device != "cpu":
        with th.cuda.device(args.device):th.cuda.empty_cache();

    with open(result_path + "history.pkl", "wb") as f:
        pickle.dump(history, f);

if __name__ == '__main__':
    args = parser.parse_args();
    main(args);
