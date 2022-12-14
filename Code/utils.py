# contains nessasary functions that are not mainly related to our project.

import torch as th
import numpy as np
import random
import os
import matplotlib.pyplot as plt

def seed_init(seed):
    th.manual_seed(seed);
    if th.cuda.is_available():
        th.cuda.manual_seed(seed);
    np.random.seed(seed);
    random.seed(seed);
    th.backends.cudnn.deterministic = True;
    th.backends.cudnn.benchmark = False;

def hms(x):
    return "{}h {:02d}m {:02d}s".format(x // 3600, x // 60 % 60, x % 60);

def get_test_val(model, test_label, device, vbatch_size):
    model = model.to(device);
    model.eval();
    with th.no_grad():
        feature_photo, feature_sketch = [], [];
        for step in range(0, test_label.shape[0], vbatch_size):
            batch_photo, batch_sketch = load_data(test_label[step: step + vbatch_size], device);
            feature_photo.append(model(batch_photo)); feature_sketch.append(model(batch_sketch));
        
        feature_photo, feature_sketch = th.cat(feature_photo, dim = 0), th.cat(feature_sketch, dim = 0);
        dist = (feature_photo.unsqueeze(1) - feature_sketch.unsqueeze(0)).pow(2).sum(2);

        diag = th.arange(dist.shape[0], device = device);
        f_dist = th.sqrt(dist[diag, diag]);
        f_avgdist, f_stddist = th.mean(f_dist).item(), th.std(f_dist).item();
        f_acc = th.mean((th.argmin(dist, 0) == diag).float()).item();

        N_DIAG = th.eye(10, dtype = bool) ^ True;
        dist[diag, diag] = th.max(dist) + 1.0;
        z_dist = th.sqrt(th.stack([dist[i: i + 10, i: i + 10][N_DIAG] for i in range(0, test_label.shape[0], 10)]));
        z_avgdist, z_stddist = th.mean(z_dist).item(), th.mean(th.std(z_dist, dim = 1)).item();
        z_acc = th.mean((th.argmin(dist, 0).div(10, rounding_mode = 'trunc') == diag.div(10, rounding_mode = 'trunc')).float()).item();
    
    return f_avgdist, f_stddist, f_acc, z_avgdist, z_stddist, z_acc;

def make_result_dir(result_path):
    result_dirs = os.listdir(result_path);
    result_idx = 0;
    while "Result{}".format(result_idx) in result_dirs: result_idx += 1;
    result_path = "{}/Result{}".format(result_path, result_idx);
    os.mkdir(result_path);
    return result_path + '/';

def plot_graph(history, result_path):
    plt.figure(figsize = (3 * 5, 4));
    plt.subplot(1, 3, 1);
    plt.title("Loss");
    plt.plot(history["epoch"], history["loss"]);
    plt.xlabel("Epoch");
    plt.ylabel("Loss");

    plt.subplot(1, 3, 2);
    plt.title("Avg. Distance (Test. Zero: {:.3f}, Few: {:.3f})".format(history["test_zz_avgdist"], history["test_ff_avgdist"]));
    plt.plot(history["epoch"], history["valid_z_avgdist"], label = "Valid Zero");
    plt.plot(history["epoch"], history["valid_f_avgdist"], label = "Valid Few");
    plt.plot(history["epoch"], history["train_avgdist"], label = "Train");
    plt.xlabel("Epoch");
    plt.ylabel("Euclidean Distance");
    plt.legend();

    plt.subplot(1, 3, 3);
    plt.title("Accuracy (Test. Zero: {:.3f}, Few: {:.3f})".format(history["test_zz_acc"], history["test_ff_acc"]));
    plt.plot(history["epoch"], history["valid_z_acc"], label = "Valid Zero");
    plt.plot(history["epoch"], history["valid_f_acc"], label = "Valid Few");
    plt.xlabel("Epoch");
    plt.ylabel("Accuracy");
    plt.legend();

    plt.savefig(result_path + "Log.jpg");
    plt.clf();
    plt.close('all');

def load_data(label, device):
    return th.from_numpy(np.stack([np.load("../Data/Preprocessed/{}_{}p.npy".format(l[0], l[1])) for l in label])).to(device), \
           th.from_numpy(np.stack([np.load("../Data/Preprocessed/{}_{}s.npy".format(l[0], l[1])) for l in label])).to(device);

def plot_train_distribution(result_path, epoch, train_distribution, train_avgdist, tau):
    result_path = result_path + "TrainDistanceDensity/";
    if epoch == 1: os.mkdir(result_path);

    sum_td = sum(train_distribution);
    train_distribution = [i / sum_td * 10.0 for i in train_distribution];

    plt.figure(figsize = (6, 4));
    plt.title("Train Distance Density (Epoch {})".format(epoch));
    plt.plot(np.arange(len(train_distribution)) / 10.0, train_distribution, color = [0, 0, 1, 1], label = "Density");
    plt.plot([train_avgdist] * 2, [0, max(train_distribution)], color = [0, 1, 0, 1], label = "Avg", linewidth = 1.0);
    plt.plot([tau] * 2, [0, max(train_distribution)], color = [1, 0, 0, 1], label = "Tau", linewidth = 0.5);
    plt.xlabel("Distance of a Pair");
    plt.ylabel("Density");
    plt.legend();

    plt.savefig(result_path + "{}.jpg".format(epoch), dpi = 300);
    plt.clf();
    plt.close('all');
