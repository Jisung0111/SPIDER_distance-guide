import torch.nn.functional as F
from torchvision.io import read_image

import numpy as np
import pickle
import os

with open("Idx_to_name.pkl", "rb") as f: idx_to_name = pickle.load(f);
with open("Idx_to_num.pkl", "rb") as f: idx_to_num = pickle.load(f);
if "Preprocessed" not in os.listdir(): os.mkdir("Preprocessed");

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
