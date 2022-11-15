# contains learning process

import torch as th
import numpy as np
import Code.neural_net as cn

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
        self.guide = guide;
        self.device = device;
        self.tau = tau;
        self.reg = reg;
        self.Q = Q;
        self.alpha = 2.0 / self.Q
        self.beta = 2.0 * self.Q
        self.gamma = -2.77 / self.Q
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

    def learn(self, epoch, train_data, valid_data):
        N = train_data[0].shape[0];
        idcs, diag = np.random.permutation(np.arange(N)), th.arange(self.batch_size);
        num_step = N // self.batch_size;
        losses, train_avgdist = 0, 0;

        self.neural_net.train();
        for step in range(num_step):
            idx = idcs[step * self.batch_size : (step + 1) * self.batch_size];
            batch_photo = th.tensor(train_data[0][idx], dtype = th.float32, device = self.device);
            batch_sketch = th.tensor(train_data[1][idx], dtype = th.float32, device = self.device);
            
            # feature_photo, feature_sketch: B * feature_dim
            feature_photo, feature_sketch = self.neural_net(batch_photo), self.neural_net(batch_sketch);
            # Squared Euclidean distance    dist[i, j]: distance between photo i, sketch j    dist: B * B
            dist = (feature_photo.unsqueeze(1) - feature_sketch.unsqueeze(0)).pow(2).sum(2);

            with th.no_grad(): train_avgdist += th.sum(dist[diag, diag]);
            
            # Loss for genuine photo and sketch pair + Loss for mismatch photo and sketch pair
            loss = self.alpha * th.sum(dist[diag, diag]) + \
                   self.beta * (th.sum(th.exp(th.sqrt(dist) * self.gamma)) - th.sum(th.exp(th.sqrt(dist[diag, diag]) * self.gamma)));

            if self.guide == 'Distance':
                # need to implement Distance guide
                #for i in range(self.batch_size):
                #    if(sqrt_dist[i, i].item() < self.tau):
                #        loss += 
                # if dist < pow(self.tau, 2): ...
                # loss = loss + self.reg * reg_loss;
                pass;
            
            losses += loss.item();

            self.optimizer.zero_grad();
            loss.backward();
            th.nn.utils.clip_grad_norm_(self.neural_net.parameters(), 1.0);
            self.optimizer.step();

            if self.lr_scheduler == 'CosineAnnealingLR':
                self.scheduler.step(epoch + (step + 1) / num_step);
        

        losses /= num_step;
        train_avgdist /= N;
        
        # check accuracy on validation set;
        self.neural_net.eval();
        with th.no_grad():
            batch_photo = th.tensor(valid_data[0], dtype = th.float32, device = self.device);
            batch_sketch = th.tensor(valid_data[1], dtype = th.float32, device = self.device);

            feature_photo, feature_sketch = self.neural_net(batch_photo), self.neural_net(batch_sketch);
            dist = (feature_photo.unsqueeze(1) - feature_sketch.unsqueeze(0)).pow(2).sum(2);

            diag = th.arange(dist.shape[0]);
            valid_avgdist = th.mean(dist[diag, diag]);
            valid_acc = th.mean(th.argmin(dist, 1) == diag);
        
        if self.lr_scheduler == 'ReduceLROnPlateau':
            self.scheduler.step(valid_avgdist);
        
        return epoch + 1, losses, train_avgdist, valid_avgdist, valid_acc;
    
    def save_model(self, file_path):
        th.save(self.neural_net.state_dict(), file_path);
