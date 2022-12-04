# contains opening preprocessed data and overall management.

import torch as th
import numpy as np
import argparse
from model import Model
import time
import json
import pickle
import utils
import os

parser = argparse.ArgumentParser();

parser.add_argument('--seed', default = 0, type = int);
parser.add_argument('--epochs', default = 80, type = int);
parser.add_argument('--batch_size', default = 100, type = int); # batch size for training set. Should divide 16000.
parser.add_argument('--vbatch_size', default = 500, type = int); # batch size for valid, test set. Should divide 2000.
parser.add_argument('--data_per_figr', default = 10, type = int);
parser.add_argument('--lr', default = 0.003, type = float);
parser.add_argument('--lr_scheduler', default = 'None', type = str); # one of [None, ReduceLROnPlateau, CosineAnnealingLR, StepLR, ExponentialLR]
parser.add_argument('--step_size', default = 40, type = int); # Step size of StepLR.
parser.add_argument('--input_size', default = '224_224', type = str); # 224 * 224 * 3 -> '224_224' # Original VGG-19 and Resnet get 224x224 input.
parser.add_argument('--batch_norm', default = 1, type = int); # indicates to use batch norm
parser.add_argument('--loss_setting', default = 2, type = int); # if loss setting is 0, general loss. or 1, only changes Y.
parser.add_argument('--feature_dim', default = 64, type = int); # dimension of output of CNN.
parser.add_argument('--guide', default = 'Distance', type = str); # 'Distance' or 'None'
parser.add_argument('--tau', default = 1.5, type = float); # used to determine necessity of distance guidance. still not sure how much value is appropriate.
parser.add_argument('--reg', default = 1.0, type = float); # weight of reg_loss. still not sure how much value is appropriate.
parser.add_argument('--Q', default = 2.5, type = float); # used for calculating hyper parameter alpha, beta, gamma.
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
        args.Q,
        args.step_size
    );

    # Make a result saving directory
    result_path = utils.make_result_dir("../Results");
    print("Saving on", result_path);
    with open(result_path + "hparam.json", 'w') as f: json.dump(vars(args), f, indent = 4);

    history = {
        "epoch": [],
        "best_f_epoch": 0,
        "best_z_epoch": 0,
        "loss": [],
        "train_avgdist": [],
        "valid_f_avgdist": [],
        "valid_f_stddist": [],
        "valid_f_acc": [],
        "valid_z_avgdist": [],
        "valid_z_stddist": [],
        "valid_z_acc": [],
        "test_ff_avgdist": 0,
        "test_ff_stddist": 0,
        "test_ff_acc": 0,
        "test_fz_avgdist": 0,
        "test_fz_stddist": 0,
        "test_fz_acc": 0,
        "test_zf_avgdist": 0,
        "test_zf_stddist": 0,
        "test_zf_acc": 0,
        "test_zz_avgdist": 0,
        "test_zz_stddist": 0,
        "test_zz_acc": 0
    }

    # Learning Process
    try:
        epoch = 0;
        while epoch < args.epochs:
            epoch, loss, train_avgdist, train_stddist, train_distribution, \
            valid_f_avgdist, valid_f_stddist, valid_f_acc, valid_z_avgdist, valid_z_stddist, valid_z_acc = \
                model.learn(epoch, labels[0], labels[1], args.vbatch_size);
            history["epoch"].append(epoch);
            history["loss"].append(loss);
            history["train_avgdist"].append(train_avgdist);
            history["valid_f_avgdist"].append(valid_f_avgdist);
            history["valid_f_stddist"].append(valid_f_stddist);
            history["valid_f_acc"].append(valid_f_acc);
            history["valid_z_avgdist"].append(valid_z_avgdist);
            history["valid_z_stddist"].append(valid_z_stddist);
            history["valid_z_acc"].append(valid_z_acc);
            with open(result_path + "Training_Log.txt", 'a') as f:
                f.write("Epoch {} ({})\tLoss: {:.4f}\tTrain Dist: {:.4f} +/- {:.4f}\tValid fDist: {:.4f} +/- {:.4f}\tValid fAcc: {:.4f}\tValid zDist: {:.4f} +/- {:.4f}\tValid zAcc: {:.4f}\n".format(
                    epoch, utils.hms(int(time.time() - start_T)), loss, train_avgdist, train_stddist, \
                    valid_f_avgdist, valid_f_stddist, valid_f_acc, valid_z_avgdist, valid_z_stddist, valid_z_acc
                ));
            if max(history["valid_f_acc"]) == valid_f_acc:
                history["best_f_epoch"] = epoch;
                model.save_model(result_path + "model.pth");
            if max(history["valid_z_acc"]) == valid_z_acc:
                history["best_z_epoch"] = epoch;
                model.save_model(result_path + "model0.pth");
            utils.plot_train_distribution(result_path, epoch, train_distribution, train_avgdist, args.tau);
            if epoch % 5 == 0:
                utils.plot_graph(history, result_path);
                with open(result_path + "history.pkl", "wb") as f: pickle.dump(history, f);
            
    except Exception as e:
        with open(result_path + "Training_Log.txt", 'a') as f:
            f.write("Accidently Training Stopped\nError Message: {}".format(repr(e)));

    # Finishing
    model.neural_net.load_state_dict(th.load(result_path + "model.pth", map_location = args.device));
    history["test_ff_avgdist"], history["test_ff_stddist"], history["test_ff_acc"], history["test_fz_avgdist"], history["test_fz_stddist"], history["test_fz_acc"] \
        = utils.get_test_val(model.neural_net, labels[2], args.device, args.vbatch_size);

    model.neural_net.load_state_dict(th.load(result_path + "model0.pth", map_location = args.device));
    history["test_zf_avgdist"], history["test_zf_stddist"], history["test_zf_acc"], history["test_zz_avgdist"], history["test_zz_stddist"], history["test_zz_acc"] \
        = utils.get_test_val(model.neural_net, labels[2], args.device, args.vbatch_size);
    
    with open(result_path + "Training_Log.txt", 'a') as f:
        f.write("\nTraining Done ({})\n".format(utils.hms(int(time.time() - start_T))));
        f.write("\nModel with Best Few Accuracy on Validation set (Epoch: {})\n".format(history["best_f_epoch"]));
        f.write("Test fDist: {:.4f} +/- {:.4f}\tTest fAcc: {:.4f}\nTest zDist: {:.4f} +/- {:.4f}\tTest zAcc: {:.4f}\n".format(
            history["test_ff_avgdist"], history["test_ff_stddist"], history["test_ff_acc"], history["test_fz_avgdist"], history["test_fz_stddist"], history["test_fz_acc"]
        ));
        f.write("\nModel with Best Zero Accuracy on Validation set (Epoch: {})\n".format(history["best_z_epoch"]));
        f.write("Test fDist: {:.4f} +/- {:.4f}\tTest fAcc: {:.4f}\nTest zDist: {:.4f} +/- {:.4f}\tTest zAcc: {:.4f}\n".format(
            history["test_zf_avgdist"], history["test_zf_stddist"], history["test_zf_acc"], history["test_zz_avgdist"], history["test_zz_stddist"], history["test_zz_acc"]
        ));

    utils.plot_graph(history, result_path);

    del model;
    if args.device != "cpu":
        with th.cuda.device(args.device):th.cuda.empty_cache();

    with open(result_path + "history.pkl", "wb") as f: pickle.dump(history, f);
    os.system("python reviewer.py --result {} --max_thres 10".format(result_path.split("Result")[-1][:-1]));

if __name__ == '__main__':
    args = parser.parse_args();
    main(args);
