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

def get_test_val(model, test_data, device):
    model = model.to(device);
    model.eval();
    with th.no_grad():
        batch_photo = th.tensor(test_data[0], dtype = th.float32, device = device);
        batch_sketch = th.tensor(test_data[1], dtype = th.float32, device = device);

        feature_photo, feature_sketch = model(batch_photo), model(batch_sketch);
        dist = (feature_photo.unsqueeze(1) - feature_sketch.unsqueeze(0)).pow(2).sum(2);

        diag = th.arange(dist.shape[0]);
        avgdist = th.mean(dist[diag, diag]);
        acc = th.mean(th.argmin(dist, 1) == diag);
    return avgdist, acc;

def make_result_dir(result_path):
    result_dirs = os.listdir(result_path);
    result_idx = 0;
    while "{}/Result{}".format(result_path, result_idx) in result_dirs: result_idx += 1;
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
