from torchvision.io import read_image

import numpy as np
import pickle

# pickle.load(f): [[dir(n001586), name(Cho_Kyuhyun), [files original]], ...]
with open("idcs.pkl", "rb") as f: idx_to_dir = {i: int(l[0][1:]) for i, l in enumerate(pickle.load(f))};

np.random.seed(1101);
name_idcs = np.random.permutation(2000);
idcs = [np.sort(name_idcs[:1600]), np.sort(name_idcs[1600:1800]), np.sort(name_idcs[1800:2000])];

for split, split_name in enumerate(["train", "valid", "test"]):
    label = [];
    for i in idcs[split]:
        dir = idx_to_dir[i];
        photo, sketch = [], [];
        for j in range(10):
            img = read_image("Photo/{}_{}.png".format(dir, j));
            img = img.float().numpy() / 255. * 2. - 1.;
            np.save("Preprocessed/{}_{}p.npy".format(i, j), img);
            
            img = read_image("Sketch/{}_{}.png".format(dir, j));
            img = img.float().numpy() / 255. * 2. - 1.;
            np.save("Preprocessed/{}_{}s.npy".format(i, j), img);
            
            label.append([i, j]);
    np.save("label_" + split_name + ".npy", label);
    print("Finished Saving {} set files".format(split_name));
