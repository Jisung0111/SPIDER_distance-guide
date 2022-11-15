# contains opening preprocessed data and overall management.

import torch as th
import numpy as np
import argparse
from Code.model import Model
import time
import json
import pickle
import utils

parser = argparse.ArgumentParser();

parser.add_argument('--seed', defalut = 0, type = int);
parser.add_argument('--epochs', default = 50, type = int);
parser.add_argument('--batch_size', default = 100, type = int);
parser.add_argument('--lr', default = 0.001, type = float);
parser.add_argument('--lr_scheduler', default = 'None', type = str); # one of [None, ReduceLROnPlateau, CosineAnnealingLR]
parser.add_argument('--input_size', default = '224_224', type = str); # 224 * 224 * 3 -> '224_224' # Original VGG-19 and Resnet get 224x224 input.
parser.add_argument('--batch_norm', default = 1, type = int); # indicates to use batch norm
parser.add_argument('--feature_dim', default = 64, type = int); # dimension of output of CNN.
parser.add_argument('--guide', default = 'Distance', type = str); # 'Distance' or 'None'
parser.add_argument('--tau', default = 0.2, type = float); # used to determine necessity of distance guidance. still not sure how much value is appropriate.
parser.add_argument('--reg', default = 0.2, type = float); # weight of reg_loss. still not sure how much valie is appropriate.
parser.add_argument('--Q', default = 10.0, type = float); # used for calculating hyper parameter alpha, beta, gamma.
parser.add_argument('--neural_net', default = 'ResNet-50', type = str); # one of {VGG-11, VGG-13, VGG-16, VGG-19, ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152}
parser.add_argument('--device', default = 'cuda:0', type = str);

datasets = ["CUFS", "CUFSF", "CUHK"];
data_split = {'train': 0, 'valid': 1, 'test': 2};
photo_sketch_idx = {"photo": 0, "sketch": 1, 'label': 2}

def main(args):
    start_T = time.time();
    utils.seed_init(args.seed);

    # Data Loading
    data_load = [
        [[np.load("Dataset/preprocessed/{}_{}_{}.npy".format(dataset, ph_sk, category)) for dataset in datasets] for category in data_split]
    for ph_sk in photo_sketch_idx]; # 3(photo, sketch, label) * 3 (train, valid, test) * # datasets
    # e.g. Dataset/preprocessed/CUHK_photo_valid.npy    ->    data_load[0][1][2]: np.array(N, 3, 224, 224)   N is the size of the validation set of CUHK dataset.
    data = [[np.concatenate(data_load[i][j], 0) for i in range(3)] for j in range(3)];     # 3(train, valid, test) * 3(photo, sketch, label)
    # e.g. data[0][1]: np.array(\sigma N, 3, 224, 224)   \sigma N is the total sum of the sketch training set sizes.

    # e.g. Dataset/preprocessed/SPI_photo_valid.npy    ->    ['Siraj's 0th photo'(3*224*244),
    #      validation set of photos in SPI dataset            'Siraj's 1st photo'(3*224*224),
    #                                                         'Chanyang's 4th photo'(3*224*224),  # Chanyang's 0th photo might be in the training set.
    #                                                         'Unidentified person's 1st photo'(3*224*244), ... ]
    # e.g. Dataset/preprocessed/SPI_sketch_valid.npy   ->    ['Siraj's 0th sketch'(3*224*244), # This matches the Siraj's 0th phooto
    #                                                         'Siraj's 1st sketch'(3*224*224),
    #                                                         'Chanyang's 4th sketch'(3*224*224),
    #                                                         'Unidentified person's 1st sketch'(3*224*244), ... ]
    # e.g. Dataset/preprocessed/SPI_label_valid.npy    ->    ['SPI|Siraj|0',
    #                                                         'SPI|Siraj|1', 
    #                                                         'SPI|Chanyang|4',
    #                                                         'SPI|noname|1', 
    #                                                         'SPI|noname|2', 
    #                                                         'SPI|Siraj|2', ... ]
    # labels are for testing same person with different poses. ('noname' will not be used for this.)
    # labels need to contain dataset name because of homonym. (assume that homonym in same dataset is already classfied well.)
    # if a dataset does not contain exact human name, then it is okay to just input distinguishable strings (e.g. H1, H2, H3, ... )

    model = Model(
        args.lr,
        args.lr_scheduler,
        args.batch_size,
        args.input_size,
        args.feature_dim,
        args.batch_norm,
        args.guide,
        args.neural_net,
        th.device(args.device),
        args.tau,
        args.reg,
        args.Q
    );

    # Make a result saving directory
    result_path = utils.make_result_dir("Results");
    with open(result_path + "hparam.json", 'w') as f: json.dump(vars(args), f, indent = 4);

    history = {
        "epoch": [],
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
            epoch, loss, train_avgdist, valid_avgdist, valid_acc = model.learn(epoch, data[0], data[1]);
            history["epoch"].append(epoch);
            history["loss"].append(loss);
            history["train_avgdist"].append(train_avgdist);
            history["valid_avgdist"].append(valid_avgdist);
            history["valid_acc"].append(valid_acc);
            with open(result_path + "Training_Log.txt", 'a') as f:
                f.write("Epoch {} ({})\tLoss: {:.4f}\tTraining set average distance: {:.4f}\tValidation set average distance: {:.4f}\tValidation set accuracy: {:.4f}\n".format(
                    epoch, utils.hms(int(time.time() - start_T)), loss, train_avgdist, valid_avgdist, valid_acc
                ));
            if max(history["valid_acc"]) == valid_acc: model.save_model(result_path + "model.pth");
    except:
        with open(result_path + "Training_Log.txt", 'a') as f:
            f.write("Accidently Training Stopped\n");

    # Finishing
    model.neural_net.load_state_dict(result_path + "model.pth", map_location = args.device);
    history["test_avgdist"], history["test_acc"] = utils.get_test_val(model.neural_net, data[2], args.device);
    with open(result_path + "Training_Log.txt", 'a') as f:
        f.write("\nTraining Done ({})\nModel with Best performance on Validation set\nTest set average distance: {:.4f}\tTest set accuracy: {:.4f}\n".format(
            utils.hms(int(time.time() - start_T)), history["test_avgdist"], history["test_acc"]
        ));

    utils.plot_graph(history, result_path);

    del model;
    if args.device != "cpu":
        with th.cuda.device(args.device):th.cuda.empty_cache();

    with open(result_path + "history.pickle", "wb") as f:
        pickle.dump(history, f);


if __name__ == '__main__':
    args = parser.parse_args();
    main(args);
