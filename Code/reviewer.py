import torch as th
import numpy as np
import argparse
from model import Model
import json
import pickle
import os

parser = argparse.ArgumentParser();
parser.add_argument('--result');
pre_args = parser.parse_args();
BATCH_SIZE = 1000; # should divide 16000 and 2000,  be divisible by 10
THRESHOLDS = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 200.0];
HITS = [1, 3, 5, 10, 20, 30, 50, 100, 200, 500, 1000];

def do_review(model, device):
    data_split = {0: 'train', 1: 'valid', 2: 'test'};
    labels = [np.load("../Data/label_{}.npy".format(split)) for split in data_split.values()];
    
    review = {};
    model = model.to(device);
    model.eval();
    
    # Implemented in this way because of Large Number of Training Set Data.
    for split in range(3):
        diag = th.arange(BATCH_SIZE);
        s_name = data_split[split] + "_";
        N = labels[split].shape[0];
        review[s_name + "max_rank"] = N;
        
        for s_i in range(0, N, BATCH_SIZE):
            batch_sketch = th.from_numpy(np.stack([np.load("../Data/Preprocessed/{}_{}s.npy".format(l[0], l[1])) for l in labels[split][s_i: s_i + BATCH_SIZE]])).to(device);
            f_photo, f_sketch = [], model(batch_sketch);
            for p_i in range(0, N, BATCH_SIZE):
                batch_photo = th.from_numpy(np.stack([np.load("../Data/Preprocessed/{}_{}p.npy".format(l[0], l[1])) for l in labels[split][p_i: p_i + BATCH_SIZE]])).to(device);
                f_photo.append(model(batch_photo));
            print("s_i", s_i);
            dist = th.sqrt((th.cat(f_photo, dim = 0).unsqueeze(1) - f_sketch.unsqueeze(0)).pow(2).sum(2));
            
            del batch_sketch, batch_photo, f_photo, f_sketch;
            if device != th.device("cpu"):
                with th.cuda.device(device): th.cuda.empty_cache();
            
            review[s_name + "avg_dist"] = review.get(s_name + "avg_dist", 0) + th.sum(dist).item();
            
            z_dist = th.cat([dist[i: i + 10, i - s_i: i - s_i + 10] for i in range(s_i, s_i + BATCH_SIZE, 10)], dim = 1);
            review[s_name + "z_avg_dist"] = review.get(s_name + "z_avg_dist", 0) + th.sum(z_dist).item();
            review[s_name + "z_std_dist"] = review.get(s_name + "z_std_dist", 0) + th.sum(th.std(z_dist.T.reshape(-1, 100), dim = 1)).item();
            
            f_dist = dist[s_i + diag, diag];
            review[s_name + "f_avg_dist"] = review.get(s_name + "f_avg_dist", 0) + th.sum(f_dist).item();
            review[s_name + "f_std_dist"] = review.get(s_name + "f_std_dist", []) + f_dist.cpu().tolist();
            
            for thres in THRESHOLDS:
                t_name = "thres{:.1f}".format(thres);
                review[s_name + t_name] = review.get(s_name + t_name, 0) + th.sum(dist <= thres).item();
                review[s_name + "z_" + t_name] = review.get(s_name + "z_" + t_name, 0) + th.sum(z_dist <= thres).item();
                review[s_name + "f_" + t_name] = review.get(s_name + "f_" + t_name, 0) + th.sum(f_dist <= thres).item();
            
            del z_dist, f_dist;
            if device != th.device("cpu"):
                with th.cuda.device(device): th.cuda.empty_cache();
            
            rank = 1 + th.argsort(th.argsort(dist, dim = 0), dim = 0);
            z_rank = th.min(th.cat([rank[i: i + 10, i - s_i: i - s_i + 10] for i in range(s_i, s_i + BATCH_SIZE, 10)], dim = 1), dim = 0)[0];
            f_rank = rank[s_i + diag, diag];
            
            for hit in HITS:
                h_name = "hit@{}".format(hit);
                review[s_name + "z_" + h_name] = review.get(s_name + "z_" + h_name, 0) + th.sum(z_rank <= hit).item();
                review[s_name + "f_" + h_name] = review.get(s_name + "f_" + h_name, 0) + th.sum(f_rank <= hit).item();
            
            z_rank = z_rank.float();
            review[s_name + "z_mr"] = review.get(s_name + "z_mr", 0) + th.sum(z_rank).item();
            review[s_name + "z_mrr"] = review.get(s_name + "z_mrr", 0) + th.sum(1.0 / z_rank).item();
            
            f_rank = f_rank.float();
            review[s_name + "f_mr"] = review.get(s_name + "f_mr", 0) + th.sum(f_rank).item();
            review[s_name + "f_mrr"] = review.get(s_name + "f_mrr", 0) + th.sum(1.0 / f_rank).item();
            
            del dist, rank, z_rank, f_rank;
            if device != th.device("cpu"):
                with th.cuda.device(device): th.cuda.empty_cache();
        
        review[s_name + "avg_dist"] = review[s_name + "avg_dist"] / (N * N);
        review[s_name + "z_avg_dist"] = review[s_name + "z_avg_dist"] / (10 * N);
        review[s_name + "z_std_dist"] = review[s_name + "z_std_dist"] / (N // 10);
        review[s_name + "f_avg_dist"] = review[s_name + "f_avg_dist"] / N;
        review[s_name + "f_std_dist"] = th.std(th.tensor(review[s_name + "f_std_dist"])).item();
        
        for hit in HITS:
            h_name = "hit@{}".format(hit);
            review[s_name + "z_" + h_name] = review[s_name + "z_" + h_name] / N;
            review[s_name + "f_" + h_name] = review[s_name + "f_" + h_name] / N;
        
        review[s_name + "z_mr"] = review[s_name + "z_mr"] / N;
        review[s_name + "z_mrr"] = review[s_name + "z_mrr"] / N;
        review[s_name + "f_mr"] = review[s_name + "f_mr"] / N;
        review[s_name + "f_mrr"] = review[s_name + "f_mrr"] / N;
    
    return review;

def main(args, result_path):
    device = th.device(args.device);
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
        device,
        args.tau,
        args.reg,
        args.Q
    );
    
    model.neural_net.load_state_dict(th.load(result_path + "model.pth", map_location = device));
    with th.no_grad(): review = do_review(model.neural_net, device);
    with open(result_path + "review.pkl", "wb") as f: pickle.dump(review, f);

if __name__ == '__main__':
    result_path = "../Results/Result" + str(pre_args.result) + "/";
    with open(result_path + "hparam.json", "r") as f:
        args = argparse.Namespace(**json.load(f));
    main(args, result_path);
