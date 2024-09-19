from copy import deepcopy
import numpy as np
import random
import torch
import os

EPS = 1e-8 

class ReplayBuffer:
    def __init__(
            self, device:torch.device, 
            n_envs:int,
            batch_size:int, 
            replay_size:int,
            obs_dim:int,
            action_dim:int) -> None:

        self.device = device
        self.n_envs = n_envs
        self.batch_size = batch_size
        self.replay_size = replay_size 
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.batch_size_per_env = int(self.batch_size/self.n_envs)
        self.replay_size_per_env = int(self.replay_size/self.n_envs)

        self.cnt = 0
        self.is_full = False
        self.obs = torch.zeros((self.replay_size_per_env, self.n_envs, self.obs_dim), device=self.device, dtype=torch.float32)
        self.actions = torch.zeros((self.replay_size_per_env, self.n_envs, self.action_dim), device=self.device, dtype=torch.float32)

    ################
    # Public Methods
    ################

    def getLen(self):
        if self.is_full:
            return self.replay_size
        else:
            return self.n_envs*self.cnt

    def addTransition(self, obs, actions):
        self.obs[self.cnt, :, :] = obs
        self.actions[self.cnt, :, :] = actions
        self.cnt += 1
        if self.cnt == self.replay_size_per_env:
            self.is_full = True
            self.cnt = 0

    @torch.no_grad()
    def getBatches(self, obs_rms):
        if self.is_full:
            random_idxs = torch.randint(0, self.replay_size_per_env, (self.batch_size_per_env,))
        else:
            random_idxs = torch.randint(0, self.cnt, (self.batch_size_per_env,))
        obs_tensor = obs_rms.normalize(self.obs[random_idxs, :, :])
        actions_tensor = self.actions[random_idxs, :, :]
        return obs_tensor.view(-1, self.obs_dim), actions_tensor.view(-1, self.action_dim)
