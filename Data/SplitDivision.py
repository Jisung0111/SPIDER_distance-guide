import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.io import read_image

import numpy as np
import os
import pickle

cnt = 0;
name_num = {};
for file_name in os.listdir("Sketch"):
    num = file_name[-12: -4];
    name = file_name[:-13];
    if name not in name_num: name_num[name] = [num];
    else: name_num[name].append(num);
    cnt += 1;

for name, num in name_num.items():
    if len(num) != 10:
        print("{} has {} Data".format(name, len(num)));
        import sys
        sys.exit();
        
print("# Name: {}, # Data for each Name: 10".format(len(name_num)));

name_to_idx = {name: i for i, name in enumerate(name_num)};
idx_to_name = {val: key for key, val in name_to_idx.items()};
num_to_idx = {name: {j: i for i, j in enumerate(name_num[name])} for name in name_num};
idx_to_num = {name: {val: key for key, val in num_to_idx[name].items()} for name in num_to_idx};

with open("Idx_to_name.pkl", "wb") as f: pickle.dump(idx_to_name, f);
with open("Idx_to_num.pkl", "wb") as f: pickle.dump(idx_to_num, f);

np.random.seed(1101);
name_idcs = np.random.permutation(2000);
idcs = [np.sort(name_idcs[:1600]), np.sort(name_idcs[1600:1800]), np.sort(name_idcs[1800:2000])];

for split, split_name in enumerate(["train", "valid", "test"]):
    label = [];
    for i in idcs[split]:
        name = idx_to_name[i];
        photo, sketch = [], [];
        for j in idx_to_num[name]:
            img = read_image("Photo/" + name + "_" + idx_to_num[name][j] + ".png");
            img = np.array(F.pad(img[:, 13: 250 - 13, :], (12, 12), "constant", 255), dtype = np.float32) / 255. * 2. - 1.;
            np.save("Preprocessed/{}_{}p.npy".format(i, j), img);
            
            img = read_image("Sketch/" + name + "_" + idx_to_num[name][j] + ".png");
            img = np.array(F.pad(img[:, 13: 250 - 13, :], (12, 12), "constant", 255), dtype = np.float32) / 255. * 2. - 1.;
            np.save("Preprocessed/{}_{}s.npy".format(i, j), img);
            
            label.append([i, j]);
    np.save("label_" + split_name + ".npy", label);
    print("Finished Saving {} set files".format(split_name));
