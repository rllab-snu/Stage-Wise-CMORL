from copy import deepcopy
import numpy as np
import pickle
import torch
import os

class ObsRMS(object):
    def __init__(self, name:str, obs_dim:int, history_len:int, device:torch.device, max_cnt=None):
        self.name = name
        self.obs_dim = obs_dim
        self.history_len = history_len
        self.max_cnt = max_cnt

        self.raw_obs_dim = int(self.obs_dim/self.history_len)
        self.mean = torch.zeros(self.raw_obs_dim, dtype=torch.float32, requires_grad=False, device=device)
        self.var = torch.ones(self.raw_obs_dim, dtype=torch.float32, requires_grad=False, device=device)
        self.count = 0

        self.cur_mean = torch.zeros_like(self.mean)
        self.cur_var = torch.ones_like(self.var)

    @torch.no_grad()
    def update(self, raw_data):
        if self.max_cnt is not None and self.count >= self.max_cnt:
            return

        data = raw_data[:, -self.raw_obs_dim:]
        count = data.shape[0]
        mean = data.mean(dim=0) # (raw_obs_dim,)
        var = data.var(dim=0, unbiased=False) # (raw_obs_dim,)
        delta_mean = mean - self.mean
        
        total_count = self.count + count
        m_a = self.var * self.count
        m_b = var * count
        M2 = m_a + m_b + delta_mean**2 * (self.count * count / total_count)
        
        self.mean += delta_mean * count / total_count
        self.var = M2 / total_count
        self.count += count

    @torch.no_grad()
    def normalize(self, observations, shifted_mean=0.0, shifted_std=1.0):
        reshaped_obs = observations.view(-1, self.obs_dim)
        reshaped_mean = self.cur_mean.view(1, -1).tile(1, self.history_len)
        reshaped_var = self.cur_var.view(1, -1).tile(1, self.history_len)
        norm_obs = (reshaped_obs - reshaped_mean)/torch.sqrt(reshaped_var + 1e-8)
        return norm_obs.view_as(observations)*shifted_std + shifted_mean
    
    def upgrade(self):
        self.cur_mean[:] = self.mean
        self.cur_var[:] = self.var
        return
    
    def load(self, save_dir, model_num):
        file_name = f"{save_dir}/{self.name}_scale/{model_num}.pkl"
        if os.path.exists(file_name):
            with open(file_name, 'rb') as f:
                mean, var, count = pickle.load(f)
            self.mean[:] = torch.tensor(mean, dtype=torch.float32, device=self.mean.device)
            self.var[:] = torch.tensor(var, dtype=torch.float32, device=self.var.device)
            self.count = count
            self.upgrade()

    def save(self, save_dir, model_num):
        file_name = f"{save_dir}/{self.name}_scale/{model_num}.pkl"
        if not os.path.exists(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))
        with open(file_name, 'wb') as f:
            mean = self.mean.cpu().numpy()
            var = self.var.cpu().numpy()
            count = self.count
            pickle.dump([mean, var, count], f)
