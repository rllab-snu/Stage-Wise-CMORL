from algos.common.actor_gaussian import ActorGaussian as Actor
from utils import cprint

from .storage import ReplayBuffer
from .normalizer import ObsRMS

import numpy as np
import torch
import os

EPS = 1e-8

class Agent:
    def __init__(self, args) -> None:
        # for base
        self.name = args.name
        self.device = args.device
        self.save_dir = args.save_dir
        self.checkpoint_dir = f"{self.save_dir}/checkpoint"
        self.obs_dim = args.obs_dim
        self.action_dim = args.action_dim
        self.action_bound_min = args.action_bound_min
        self.action_bound_max = args.action_bound_max
        self.n_steps = args.n_steps
        self.n_envs = args.n_envs
        self.batch_size = args.batch_size
        self.replay_size = args.replay_size

        # for normalization
        self.history_len = args.history_len
        self.obs_rms = ObsRMS('obs', self.obs_dim, self.history_len, self.device)

        # for learning
        self.actor_lr = args.actor_lr
        self.n_actor_iters = args.n_actor_iters
        self.max_grad_norm = args.max_grad_norm

        # declare actor
        model_cfg = args.model
        self.actor = Actor(
            self.device, self.obs_dim, self.action_dim, self.action_bound_min, 
            self.action_bound_max, model_cfg['actor']).to(self.device)

        # declare optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        # declare replay buffer
        self.replay_buffer = ReplayBuffer(
            self.device, self.n_envs, self.batch_size, 
            self.replay_size, self.obs_dim, self.action_dim)

    ################
    # Public Methods
    ################

    @torch.no_grad()
    def getAction(self, obs_tensor:torch.tensor, deterministic:bool) -> torch.tensor:
        norm_obs_tensor = self.obs_rms.normalize(obs_tensor)
        epsilon_tensor = torch.randn(norm_obs_tensor.shape[:-1] + (self.action_dim,), device=self.device)
        self.actor.updateActionDist(norm_obs_tensor, epsilon_tensor)
        _, unnorm_action_tensor = self.actor.sample(deterministic)
        return unnorm_action_tensor

    def step(self, obs_tensor, actions_tensor):
        self.replay_buffer.addTransition(obs_tensor, actions_tensor)
    
    def copyObsRMS(self, obs_rms):
        self.obs_rms.mean[:] = obs_rms.mean
        self.obs_rms.var[:] = obs_rms.var
        self.obs_rms.upgrade()

    def readyToTrain(self):
        return True

    def train(self):
        for _ in range(self.n_actor_iters):
            obs_tensor, actions_tensor = self.replay_buffer.getBatches(self.obs_rms)
            action_means_tensor, _, _ = self.actor(obs_tensor)
            actor_loss = (action_means_tensor - actions_tensor).square().mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

        # return
        train_results = {
            'actor_loss': actor_loss.item(),
        }
        return train_results
    
    def save(self, model_num):
        # save rms
        self.obs_rms.save(self.save_dir, model_num)

        # save network models
        save_dict = {
            'actor': self.actor.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
        }
        torch.save(save_dict, f"{self.checkpoint_dir}/model_{model_num}.pt")
        cprint(f'[{self.name}] save success.', bold=True, color="blue")

    def load(self, model_num):
        # load rms
        self.obs_rms.load(self.save_dir, model_num)

        # load network models
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        checkpoint_file = f"{self.checkpoint_dir}/model_{model_num}.pt"
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            cprint(f'[{self.name}] load success.', bold=True, color="blue")
            return int(model_num)
        else:
            self.actor.initialize()
            cprint(f'[{self.name}] load fail.', bold=True, color="red")
            return 0

    ################
    # private method
    ################
