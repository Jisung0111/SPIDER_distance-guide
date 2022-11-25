# contains learning process

import torch as th
import numpy as np
import neural_net as cn
import utils

Neural_Networks = {
    "VGG-11": cn.VGG_11,
    "VGG-13": cn.VGG_13,
    "VGG-16": cn.VGG_16,
    "VGG-19": cn.VGG_19,
    "ResNet-18": cn.ResNet_18,
    "ResNet-34": cn.ResNet_34,
    "ResNet-50": cn.ResNet_50,
    "ResNet-101": cn.ResNet_101,
    "ResNet-152": cn.ResNet_152
}


class Model:
    def __init__(
        self,
        lr,
        lr_scheduler,
        batch_size,
        data_per_figr,
        input_size,
        feature_dim,
        batch_norm,
        guide,
        neural_net,
        device,
        tau,
        reg,
        Q
    ):
        self.batch_size = batch_size;
        self.data_per_figr = data_per_figr;
        if data_per_figr % 2 == 1:
            raise ValueError("Data per Face should be EVEN.");
        
        self.guide = guide;
        self.device = device;
        self.tau = tau;
        self.reg = reg;
        self.Q = Q;
        self.alpha = 2.0 / self.Q;
        self.beta = 2.0 * self.Q;
        self.gamma = -2.77 / self.Q;
        self.neural_net = Neural_Networks[neural_net](input_size, feature_dim, batch_norm).to(device);
        self.optimizer = th.optim.Adam(self.neural_net.parameters(), lr = lr);
        
        self.lr_scheduler = lr_scheduler;
        if lr_scheduler == 'None': pass
        elif lr_scheduler == 'ReduceLROnPlateau':
            self.scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 'min', factor = 0.1, patience = 10, min_lr = 1e-5); # according to the valid euclidean distance
        elif lr_scheduler == 'CosineAnnealingLR':
            self.scheduler = th.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max = 10);
        else: raise ValueError("Wrong learning rate scheduler");
        self.y1_idx = th.eye(self.batch_size, device = self.device);
        self.y0_idx = th.triu(th.ones((batch_size, batch_size), device = device)) - self.y1_idx;

    def learn(self, epoch, train_label, valid_label, vbatch_size):
        N = train_label.shape[0];
        hN, hB, num_step = N // 2, self.batch_size // 2, N // self.batch_size;
        idcs, diag = np.random.permutation(hN), th.arange(self.batch_size);
        pair_idcs = np.array([i + np.random.permutation(self.data_per_figr) for i in range(0, N, self.data_per_figr)]).reshape((hN, 2));
        losses, train_avgdist = 0, 0;

        self.neural_net.train();
        for step in range(num_step):
            # batch_photo, batch_sketch: B * 3 * 224 * 224
            batch_photo, batch_sketch = utils.load_data(train_label[pair_idcs[idcs[step * hB: (step + 1) * hB]].reshape(-1)], self.device);
            # feature_photo, feature_sketch: B * feature_dim
            feature_photo, feature_sketch = self.neural_net(batch_photo), self.neural_net(batch_sketch);
            # Squared Euclidean distance    dist[i, j]: distance between photo i, sketch j    dist: B * B
            dist = (feature_photo.unsqueeze(1) - feature_sketch.unsqueeze(0)).pow(2).sum(2);
            sqrt_dist = th.sqrt(dist); diag_dist = sqrt_dist[diag, diag];
            
            with th.no_grad(): train_avgdist += th.sum(diag_dist);

            Y1_calc_idx, Y0_calc_idx = self.y1_idx.clone(), self.y0_idx.clone();
            if self.guide == 'Distance':
                reg_idx = th.argwhere(diag_dist < self.tau).view(-1);
                ry0_idx = th.sort(th.stack((reg_idx, reg_idx ^ 1)), dim = 0)[0];
                Y1_calc_idx[reg_idx, reg_idx] = 0.0; Y1_calc_idx[ry0_idx[0], ry0_idx[1]] = 1.0;
                Y0_calc_idx[reg_idx, reg_idx] = 1.0; Y0_calc_idx[ry0_idx[0], ry0_idx[1]] = 0.0;
            
            loss = (self.alpha * th.sum(Y1_calc_idx * dist) + self.beta * th.sum(Y0_calc_idx * th.exp(sqrt_dist * self.gamma)));
            loss = loss / ((self.batch_size * self.batch_size + self.batch_size) / 2);
            
            losses += loss.item();

            self.optimizer.zero_grad();
            loss.backward();
            th.nn.utils.clip_grad_norm_(self.neural_net.parameters(), 1.0);
            self.optimizer.step();

            if self.lr_scheduler == 'CosineAnnealingLR':
                self.scheduler.step(epoch + (step + 1) / num_step);
        
        if self.device != th.device("cpu"):
            with th.cuda.device(self.device): th.cuda.empty_cache();

        losses /= num_step;
        train_avgdist /= N;
        
        # check accuracy on validation set;
        self.neural_net.eval();
        with th.no_grad():
            num_step = valid_label.shape[0] // vbatch_size;
            feature_photo, feature_sketch = [], [];
            for step in range(num_step):
                batch_photo, batch_sketch = utils.load_data(valid_label[step * vbatch_size: (step + 1) * vbatch_size], self.device);
                feature_photo.append(self.neural_net(batch_photo)); feature_sketch.append(self.neural_net(batch_sketch));
            
            feature_photo, feature_sketch = th.cat(feature_photo, dim = 0), th.cat(feature_sketch, dim = 0);
            dist = (feature_photo.unsqueeze(1) - feature_sketch.unsqueeze(0)).pow(2).sum(2);

            diag = th.arange(dist.shape[0]);
            valid_avgdist = th.mean(th.sqrt(dist[diag, diag]));
            valid_acc = th.mean((th.argmin(dist, 1).cpu() == diag).float());
        
        if self.device != th.device("cpu"):
            with th.cuda.device(self.device): th.cuda.empty_cache();
        
        if self.lr_scheduler == 'ReduceLROnPlateau':
            self.scheduler.step(valid_avgdist);
        
        return epoch + 1, losses, train_avgdist, valid_avgdist, valid_acc;
    
    def save_model(self, file_path):
        th.save(self.neural_net.state_dict(), file_path);
