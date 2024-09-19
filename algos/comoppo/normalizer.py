from copy import deepcopy
import numpy as np
import pickle
import torch
import os

class RewardRMS(object):
    def __init__(self, name:str, reward_dim:int, num_stages:int, device:torch.device, max_cnt=None):
        self.name = name
        self.reward_dim = reward_dim
        self.num_stages = num_stages
        self.device = device
        self.max_cnt = max_cnt
        self.mean = torch.zeros((self.num_stages, self.reward_dim), dtype=torch.float32, requires_grad=False, device=self.device)
        self.var = torch.ones((self.num_stages, self.reward_dim), dtype=torch.float32, requires_grad=False, device=self.device)
        self.count = torch.zeros((self.num_stages,), dtype=torch.int64, requires_grad=False, device=self.device)

    @torch.no_grad()
    def update(self, data, stage):
        reshaped_data = data.view((-1, self.reward_dim))
        reshaped_stage = stage.view((-1, self.num_stages))
        for stage_idx in range(self.num_stages):
            if self.max_cnt is not None and self.count[stage_idx] >= self.max_cnt:
                continue

            stage_data = reshaped_data[reshaped_stage[:, stage_idx] == 1] # (N, reward_dim)
            if stage_data.numel() == 0:
                continue

            stage_count = stage_data.size(0)
            stage_mean = stage_data.mean(dim=0) # (reward_dim,)
            stage_var = stage_data.var(dim=0, unbiased=False) # (reward_dim,)
            delta_mean = stage_mean - self.mean[stage_idx] # (reward_dim,)
            
            total_count = self.count[stage_idx] + stage_count
            m_a = self.var[stage_idx] * self.count[stage_idx]
            m_b = stage_var * stage_count
            M2 = m_a + m_b + delta_mean**2 * (self.count[stage_idx] * stage_count / total_count)
            
            self.mean[stage_idx] += delta_mean * stage_count / total_count
            self.var[stage_idx] = M2 / total_count
            self.count[stage_idx] += stage_count

    @torch.no_grad()
    def normalize(self, data, stage, default_mean=0.0, default_std=1.0):
        reshaped_data = data.view(-1, self.reward_dim)
        reshaped_stage = stage.view(-1, self.num_stages)
        norm_data = torch.zeros_like(reshaped_data)
        for stage_idx in range(self.num_stages):
            mask = reshaped_stage[:, stage_idx:(stage_idx+1)] # (N, 1)
            mean = self.mean[stage_idx:(stage_idx+1)] # (1, reward_dim)
            std = torch.sqrt(self.var[stage_idx:(stage_idx+1)] + 1e-8) # (1, reward_dim)
            norm_data += mask * ((reshaped_data - mean) / std)
        return norm_data.view_as(data)*default_std + default_mean
    
    def load(self, save_dir, model_num):
        file_name = f"{save_dir}/{self.name}_scale/{model_num}.pkl"
        if os.path.exists(file_name):
            with open(file_name, 'rb') as f:
                mean, var, count = pickle.load(f)
            self.mean[:] = torch.tensor(mean, dtype=torch.float32, device=self.device)
            self.var[:] = torch.tensor(var, dtype=torch.float32, device=self.device)
            self.count[:] = torch.tensor(count, dtype=torch.int64, device=self.device)

    def save(self, save_dir, model_num):
        file_name = f"{save_dir}/{self.name}_scale/{model_num}.pkl"
        if not os.path.exists(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))
        with open(file_name, 'wb') as f:
            mean = self.mean.cpu().numpy()
            var = self.var.cpu().numpy()
            count = self.count.cpu().numpy()
            pickle.dump([mean, var, count], f)


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
