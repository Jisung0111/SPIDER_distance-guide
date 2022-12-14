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
        loss_setting,
        Q,
        step_size = 20
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
                self.optimizer, 'max', factor = 0.1, patience = 6, min_lr = 2e-6); # according to the valid euclidean distance
        elif lr_scheduler == 'CosineAnnealingLR':
            self.scheduler = th.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max = 10);
        elif lr_scheduler == 'StepLR':
            self.scheduler = th.optim.lr_scheduler.StepLR(self.optimizer, step_size = step_size, gamma = 0.1);
        elif lr_scheduler == 'ExponentialLR':
            self.scheduler = th.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma = 0.945);
        else: raise ValueError("Wrong learning rate scheduler");

        self.loss_setting = loss_setting;
        if loss_setting == 1 or loss_setting == 3:
            self.y1_idx = th.eye(self.batch_size, device = self.device);
            self.y0_idx = th.ones((batch_size, batch_size), device = device) - self.y1_idx;

    def learn(self, epoch, train_label, valid_label, vbatch_size):
        N = train_label.shape[0];
        hN, hB, num_step = N // 2, self.batch_size // 2, N // self.batch_size;
        idcs, diag = np.random.permutation(hN), th.arange(self.batch_size);
        pair_idcs = np.array([i + np.random.permutation(self.data_per_figr) for i in range(0, N, self.data_per_figr)]).reshape((hN, 2));
        losses, train_dists = 0, [];
        N_DIAG = th.eye(10, dtype = bool) ^ True;

        self.neural_net.train();
        for step in range(num_step):
            # batch_photo, batch_sketch: B * 3 * 224 * 224
            batch_photo, batch_sketch = utils.load_data(train_label[pair_idcs[idcs[step * hB: (step + 1) * hB]].reshape(-1)], self.device);
            # feature_photo, feature_sketch: B * feature_dim
            feature_photo, feature_sketch = self.neural_net(batch_photo), self.neural_net(batch_sketch);
            # Squared Euclidean distance    dist[i, j]: distance between photo i, sketch j    dist: B * B
            dist = (feature_photo.unsqueeze(1) - feature_sketch.unsqueeze(0)).pow(2).sum(2);
            sqrt_dist = th.sqrt(dist); diag_dist = sqrt_dist[diag, diag];
            
            with th.no_grad(): train_dists += diag_dist.detach().clone().cpu().tolist();

            if self.loss_setting == 1:
                Y1_calc_idx, Y0_calc_idx = self.y1_idx.clone(), self.y0_idx.clone();
                if self.guide == 'Distance':
                    reg_idx = th.argwhere(diag_dist < self.tau).view(-1);
                    Y1_calc_idx[reg_idx, reg_idx] = 0.0; Y1_calc_idx[reg_idx ^ 1, reg_idx] = 1.0;
                    Y0_calc_idx[reg_idx, reg_idx] = 1.0; Y0_calc_idx[reg_idx ^ 1, reg_idx] = 0.0;
                
                loss = self.alpha * th.sum(Y1_calc_idx * dist) + self.beta * th.sum(Y0_calc_idx * th.exp(sqrt_dist * self.gamma));
            
            elif self.loss_setting == 2:
                loss = self.alpha * th.sum(dist[diag, diag]) + \
                       self.beta * (th.sum(th.exp(dist * self.gamma)) - th.sum(th.exp(dist[diag, diag] * self.gamma)));
                
                if self.guide == 'Distance':
                    reg_idx = th.argwhere(diag_dist < self.tau).view(-1);
                    loss = loss + (self.reg * self.alpha) * th.sum(dist[reg_idx ^ 1, reg_idx]);
            
            elif self.loss_setting == 3:
                Y1_calc_idx, Y0_calc_idx = self.y1_idx.clone(), self.y0_idx.clone();
                if self.guide == 'Distance':
                    reg_idx = th.argwhere(diag_dist < self.tau).view(-1);
                    Y1_calc_idx[reg_idx, reg_idx] = 0.0; Y1_calc_idx[reg_idx ^ 1, reg_idx] = 1.0;
                    Y0_calc_idx[reg_idx, reg_idx] = 1.0; Y0_calc_idx[reg_idx ^ 1, reg_idx] = 0.0;
                
                loss = self.alpha * th.sum(Y1_calc_idx * dist) + self.beta * th.sum(Y0_calc_idx * th.exp(dist * self.gamma));

            else:
                loss = self.alpha * th.sum(dist[diag, diag]) + \
                       self.beta * (th.sum(th.exp(sqrt_dist * self.gamma)) - th.sum(th.exp(diag_dist * self.gamma)));
                
                if self.guide == 'Distance':
                    reg_idx = th.argwhere(diag_dist < self.tau).view(-1);
                    loss = loss + (self.reg * self.alpha) * th.sum(dist[reg_idx ^ 1, reg_idx]);
                
            loss = loss / (self.batch_size * self.batch_size);
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
        train_avgdist, train_stddist = np.mean(train_dists), np.std(train_dists);
        train_distribution = [0 for _ in range(int(10 * np.max(train_dists)) + 2)];
        for d in train_dists: train_distribution[int(10 * d)] += 1;
        
        # check accuracy on validation set;
        self.neural_net.eval();
        with th.no_grad():
            feature_photo, feature_sketch = [], [];
            for step in range(0, valid_label.shape[0], vbatch_size):
                batch_photo, batch_sketch = utils.load_data(valid_label[step: step + vbatch_size], self.device);
                feature_photo.append(self.neural_net(batch_photo)); feature_sketch.append(self.neural_net(batch_sketch));
            
            feature_photo, feature_sketch = th.cat(feature_photo, dim = 0), th.cat(feature_sketch, dim = 0);
            dist = (feature_photo.unsqueeze(1) - feature_sketch.unsqueeze(0)).pow(2).sum(2);

            diag = th.arange(dist.shape[0], device = self.device);
            valid_f_dist = th.sqrt(dist[diag, diag]);
            valid_f_avgdist, valid_f_stddist = th.mean(valid_f_dist).item(), th.std(valid_f_dist).item();
            valid_f_acc = th.mean((th.argmin(dist, 0) == diag).float()).item();

            dist[diag, diag] = th.max(dist) + 1.0;
            valid_z_dist = th.sqrt(th.stack([dist[i: i + 10, i: i + 10][N_DIAG] for i in range(0, dist.shape[0], 10)]));
            valid_z_avgdist, valid_z_stddist = th.mean(valid_z_dist).item(), th.mean(th.std(valid_z_dist, dim = 1)).item();
            valid_z_acc = th.mean((th.argmin(dist, 0).div(10, rounding_mode = 'trunc') == diag.div(10, rounding_mode = 'trunc')).float()).item();
        
        if self.device != th.device("cpu"):
            with th.cuda.device(self.device): th.cuda.empty_cache();
        
        if self.lr_scheduler == 'ReduceLROnPlateau':
            self.scheduler.step(valid_f_acc);
        elif self.lr_scheduler == 'StepLR':
            self.scheduler.step();
        elif self.lr_scheduler == 'ExponentialLR':
            self.scheduler.step();
        
        return epoch + 1, losses, train_avgdist, train_stddist, train_distribution, \
               valid_f_avgdist, valid_f_stddist, valid_f_acc, valid_z_avgdist, valid_z_stddist, valid_z_acc;
    
    def save_model(self, file_path):
        th.save(self.neural_net.state_dict(), file_path);
