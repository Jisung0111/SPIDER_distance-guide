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

        diag = th.arange(dist.shape[0]);
        avgdist = th.mean(th.sqrt(dist[diag, diag])).item();
        acc = th.mean((th.argmin(dist, 0).cpu() == diag).float()).item();
    
    return avgdist, acc;

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
    plt.title("Avg. Distance (Test: {:.4f})".format(history["test_avgdist"]));
    plt.plot(history["epoch"], history["train_avgdist"], label = "Train");
    plt.plot(history["epoch"], history["valid_avgdist"], label = "Valid");
    plt.xlabel("Epoch");
    plt.ylabel("Squared Euclidean Distance");
    plt.legend();

    plt.subplot(1, 3, 3);
    plt.title("Accuracy (Test: {:.4f})".format(history["test_acc"]));
    plt.plot(history["epoch"], history["valid_acc"], label = "Valid");
    plt.xlabel("Epoch");
    plt.ylabel("Accuracy");
    plt.legend();

    plt.savefig(result_path + "Log.jpg");

def load_data(label, device):
    return th.from_numpy(np.stack([np.load("../Data/Preprocessed/{}_{}p.npy".format(l[0], l[1])) for l in label])).to(device), \
           th.from_numpy(np.stack([np.load("../Data/Preprocessed/{}_{}s.npy".format(l[0], l[1])) for l in label])).to(device);
