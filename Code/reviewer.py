import torch as th
import numpy as np
import argparse
from model import Model
import json
import pickle
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser();
parser.add_argument('--result');
parser.add_argument('--max_thres', default = 10, type = float);
pre_args = parser.parse_args();
THRESHOLDS = [i for i in np.linspace(0, pre_args.max_thres, 101)];
GAP = THRESHOLDS[1]; # means gap of THRESHOLDS
HITS = [1, 3, 5, 10, 20, 30, 50, 100, 200, 500, 1000];

def do_review(model, device, reject):
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
        N_DIAG = th.eye(10, dtype = bool) ^ True;
        
        for s_i in range(0, N, BATCH_SIZE):
            batch_sketch = th.from_numpy(np.stack([np.load("../Data/Preprocessed/{}_{}s.npy".format(l[0], l[1])) for l in labels[split][s_i: s_i + BATCH_SIZE]])).to(device);
            f_photo, f_sketch = [], model(batch_sketch);
            for p_i in range(0, N, BATCH_SIZE):
                batch_photo = th.from_numpy(np.stack([np.load("../Data/Preprocessed/{}_{}p.npy".format(l[0], l[1])) for l in labels[split][p_i: p_i + BATCH_SIZE]])).to(device);
                f_photo.append(model(batch_photo));
            dist = th.sqrt((th.cat(f_photo, dim = 0).unsqueeze(1) - f_sketch.unsqueeze(0)).pow(2).sum(2));
            
            del batch_sketch, batch_photo, f_photo, f_sketch;
            if device != th.device("cpu"):
                with th.cuda.device(device): th.cuda.empty_cache();
            
            review[s_name + "avg_dist"] = review.get(s_name + "avg_dist", 0) + th.sum(dist).item();
            
            z_dist = th.cat([dist[i: i + 10, i - s_i: i - s_i + 10][N_DIAG].view((9, 10)) if reject else \
                             dist[i: i + 10, i - s_i: i - s_i + 10] for i in range(s_i, s_i + BATCH_SIZE, 10)], dim = 1);
            review[s_name + "z_avg_dist"] = review.get(s_name + "z_avg_dist", 0) + th.sum(z_dist).item();
            review[s_name + "z_std_dist"] = review.get(s_name + "z_std_dist", 0) + th.sum(th.std(z_dist.T.reshape(-1, 90 if reject else 100), dim = 1)).item();
            
            f_dist = dist[s_i + diag, diag];
            review[s_name + "f_avg_dist"] = review.get(s_name + "f_avg_dist", 0) + th.sum(f_dist).item();
            review[s_name + "f_std_dist"] = review.get(s_name + "f_std_dist", []) + f_dist.cpu().tolist();
            
            for thres in THRESHOLDS:
                t_name = "thres{:.4f}".format(thres);
                review[s_name + t_name] = review.get(s_name + t_name, 0) + th.sum(dist <= thres).item();
                review[s_name + "z_" + t_name] = review.get(s_name + "z_" + t_name, 0) + th.sum(z_dist <= thres).item();
                review[s_name + "f_" + t_name] = review.get(s_name + "f_" + t_name, 0) + th.sum(f_dist <= thres).item();
            
            del z_dist, f_dist;
            if device != th.device("cpu"):
                with th.cuda.device(device): th.cuda.empty_cache();
            
            rank = 1 + th.argsort(th.argsort(dist, dim = 0), dim = 0);
            f_rank = rank[s_i + diag, diag];

            if reject:
                dist[s_i + diag, diag] = th.max(dist) + 1.0;
                rank = 1 + th.argsort(th.argsort(dist, dim = 0), dim = 0);
            z_rank = th.min(th.cat([rank[i: i + 10, i - s_i: i - s_i + 10] for i in range(s_i, s_i + BATCH_SIZE, 10)], dim = 1), dim = 0)[0];
            
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
        review[s_name + "z_avg_dist"] = review[s_name + "z_avg_dist"] / ((9 if reject else 10) * N);
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
    print("Review", result_path);
    
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
        args.loss_setting,
        args.Q
    );

    color = [[[1.0, 0.1, 0.1, 1.0], [0.8, 0.2, 0.2, 1.0], [0.7, 0.4, 0.4, 1.0]],
             [[0.1, 1.0, 0.1, 1.0], [0.2, 0.8, 0.2, 1.0], [0.4, 0.7, 0.4, 1.0]],
             [[0.1, 0.1, 1.0, 1.0], [0.2, 0.2, 0.8, 1.0], [0.4, 0.4, 0.7, 1.0]]];
    smooth = lambda x: [x[i] if i < 3 or len(x) - 4 < i else sum([x[j] for j in range(i - 3, i + 4)]) / 7 for i in range(len(x))];
    
    for zero_set in ["", "0"]:
        model.neural_net.load_state_dict(th.load("{}model{}.pth".format(result_path, zero_set), map_location = device));
        with th.no_grad(): review = do_review(model.neural_net, device, True);
        with open("{}review{}.pkl".format(result_path, zero_set), "wb") as f: pickle.dump(review, f);

        plt.figure(figsize = (24, 14));

        plt.subplot(3, 4, 1);
        plt.title("MR (Test. Zero: {:.1f}, Few: {:.1f})".format(review["test_z_mr"], review["test_f_mr"]));
        plt.ylabel("Mean Rank");
        plt.plot(["Train", "Valid", "Test"], [review["{}_z_mr".format(split)] for split in ["train", "valid", "test"]], label = "MR Zero");
        plt.plot(["Train", "Valid", "Test"], [review["{}_f_mr".format(split)] for split in ["train", "valid", "test"]], label = "MR Few");
        plt.legend();

        plt.subplot(3, 4, 2);
        plt.title("MRR (Test. Zero: {:.4f}, Few: {:.4f})".format(review["test_z_mrr"], review["test_f_mrr"]));
        plt.ylabel("Mean Reciprocal Rank");
        plt.plot(["Train", "Valid", "Test"], [review["{}_z_mrr".format(split)] for split in ["train", "valid", "test"]], label = "MRR Zero");
        plt.plot(["Train", "Valid", "Test"], [review["{}_f_mrr".format(split)] for split in ["train", "valid", "test"]], label = "MRR Few");
        plt.legend();

        plt.subplot(3, 4, 3);
        plt.title("Accuracy (Test. Zero: {:.4f}, Few: {:.4f})".format(review["test_z_hit@1"], review["test_f_hit@1"]));
        plt.ylabel("Accuracy");
        plt.plot(["Train", "Valid", "Test"], [review["{}_z_hit@1".format(split)] for split in ["train", "valid", "test"]], label = "Accuracy Zero");
        plt.plot(["Train", "Valid", "Test"], [review["{}_f_hit@1".format(split)] for split in ["train", "valid", "test"]], label = "Accuracy Few");
        plt.legend();

        plt.subplot(3, 4, 4);
        plt.title("Hit @ 5 (Test. Zero: {:.4f}, Few: {:.4f})".format(review["test_z_hit@5"], review["test_f_hit@5"]));
        plt.ylabel("Hit @ 5");
        plt.plot(["Train", "Valid", "Test"], [review["{}_z_hit@5".format(split)] for split in ["train", "valid", "test"]], label = "Hit@5 Zero");
        plt.plot(["Train", "Valid", "Test"], [review["{}_f_hit@5".format(split)] for split in ["train", "valid", "test"]], label = "Hit@5 Few");
        plt.legend();

        plt.subplot(3, 2, 3);
        plt.title("Distance Cumulative Density");
        plt.xlabel("d");
        plt.ylabel("Ratio of Photos with Dist. <= d");
        max_val = 0;
        for idx, (s_name, split) in enumerate(zip(['train_', 'valid_', 'test_'], ["Train", "Valid", "Test"])):
            N = review[s_name + "max_rank"];
            arr = smooth([review[s_name + "z_thres{:.4f}".format(thres)] / N / 9 for thres in THRESHOLDS]);
            plt.plot(THRESHOLDS, arr, label = split + " Zero", color = color[idx][0]);
            max_val = max(max_val, max(arr));

            arr = smooth([review[s_name + "f_thres{:.4f}".format(thres)] / N for thres in THRESHOLDS]);
            plt.plot(THRESHOLDS, arr, label = split + " Few", color = color[idx][1]);
            max_val = max(max_val, max(arr));

            arr = smooth([review[s_name + "thres{:.4f}".format(thres)] / (N * N) for thres in THRESHOLDS]);
            plt.plot(THRESHOLDS, arr, label = split + " ALL", color = color[idx][2]);
            max_val = max(max_val, max(arr));

        for idx, s_name in enumerate(['train_', 'valid_', 'test_']):
            plt.plot([review[s_name + "z_avg_dist"]] * 2, [0, max_val], color = color[idx][0]);
            plt.plot([review[s_name + "f_avg_dist"]] * 2, [0, max_val], color = color[idx][1]);
            plt.plot([review[s_name + "avg_dist"]] * 2, [0, max_val], color = color[idx][2]);
        plt.legend();

        plt.subplot(3, 2, 4);
        plt.title("Distance Density (Test. Zero: {:.1f} +/- {:.1f}, Few: {:.1f} +/- {:.1f}, ALL: {:.1f})".format(
            review["test_z_avg_dist"], review["test_z_std_dist"], review["test_f_avg_dist"], review["test_f_std_dist"], review["test_avg_dist"]));
        plt.xlabel("d");
        plt.ylabel("Ratio of Photos with d-{:.2f} < Dist. <= d".format(GAP));
        max_val = 0;
        for idx, (s_name, split) in enumerate(zip(['train_', 'valid_', 'test_'], ["Train", "Valid", "Test"])):
            N = review[s_name + "max_rank"];
            arr = smooth([(review[s_name + "z_thres{:.4f}".format(thres)] - review.get(s_name + "z_thres{:.4f}".format(thres - GAP), 0)) / N / 9.0 / GAP for thres in THRESHOLDS]);
            plt.plot(THRESHOLDS, arr, label = split + " Zero", color = color[idx][0]);
            max_val = max(max_val, max(arr));

            arr = smooth([(review[s_name + "f_thres{:.4f}".format(thres)] - review.get(s_name + "f_thres{:.4f}".format(thres - GAP), 0)) / N / GAP for thres in THRESHOLDS]);
            plt.plot(THRESHOLDS, arr, label = split + " Few", color = color[idx][1]);
            max_val = max(max_val, max(arr));

            arr = smooth([(review[s_name + "thres{:.4f}".format(thres)] - review.get(s_name + "thres{:.4f}".format(thres - GAP), 0)) / (N * N) / GAP for thres in THRESHOLDS]);
            plt.plot(THRESHOLDS, arr, label = split + " ALL", color = color[idx][2]);
            max_val = max(max_val, max(arr));

        for idx, s_name in enumerate(['train_', 'valid_', 'test_']):
            plt.plot([review[s_name + "z_avg_dist"]] * 2, [0, max_val], color = color[idx][0]);
            plt.plot([review[s_name + "f_avg_dist"]] * 2, [0, max_val], color = color[idx][1]);
            plt.plot([review[s_name + "avg_dist"]] * 2, [0, max_val], color = color[idx][2]);
        plt.legend();

        ax = plt.subplot(3, 1, 3);
        ax.set_title("Hit@K");
        ax.set_xlabel("K");
        ax.set_ylabel("Hit@K");
        x = np.arange(len(HITS));

        bar_width = 0.15;
        ax.bar(x - 2.5 * bar_width, [review["train_z_hit@{}".format(hit)] for hit in HITS], width = bar_width * 0.9, label = "Train Zero", color = color[0][0]);
        ax.bar(x - 1.5 * bar_width, [review["train_f_hit@{}".format(hit)] for hit in HITS], width = bar_width * 0.9, label = "Train Few", color = color[0][1]);
        ax.bar(x - 0.5 * bar_width, [review["valid_z_hit@{}".format(hit)] for hit in HITS], width = bar_width * 0.9, label = "Valid Zero", color = color[1][0]);
        ax.bar(x + 0.5 * bar_width, [review["valid_f_hit@{}".format(hit)] for hit in HITS], width = bar_width * 0.9, label = "Valid Few", color = color[1][1]);
        ax.bar(x + 1.5 * bar_width, [review["test_z_hit@{}".format(hit)] for hit in HITS], width = bar_width * 0.9, label = "Test Zero", color = color[2][0]);
        ax.bar(x + 2.5 * bar_width, [review["test_f_hit@{}".format(hit)] for hit in HITS], width = bar_width * 0.9, label = "Test Few", color = color[2][1]);

        ax.legend();
        ax.set_xticks(x);
        ax.set_xticklabels(HITS);

        for bar in ax.patches:
            bar_value = bar.get_height();
            text = "{:.3f}".format(bar_value);
            text_x = bar.get_x() +  bar.get_width() / 2;
            text_y = bar.get_y() + bar_value;
            bar_color = bar.get_facecolor();
            ax.text(text_x, text_y, text, ha = "center", va = "bottom", color = bar_color, size = 6);

        plt.savefig("{}review{}.jpg".format(result_path, zero_set), dpi = 300);

if __name__ == '__main__':
    result_path = "../Results/Result" + str(pre_args.result) + "/";
    with open(result_path + "hparam.json", "r") as f:
        args = argparse.Namespace(**json.load(f));
    BATCH_SIZE = 400 if args.neural_net[:3] == "VGG" else 1000;
    main(args, result_path);
