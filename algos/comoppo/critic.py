from algos.common.critic_base import CriticS
from algos.common.network_base import MLP

import numpy as np
import torch


class Critic(CriticS):
    def __init__(self, device:torch.device, obs_dim:int, state_dim:int, stage_dim:int, reward_dim:int, critic_cfg:dict) -> None:
        self.state_dim = state_dim
        self.stage_dim = stage_dim
        self.reward_dim = reward_dim
        super().__init__(device, obs_dim, critic_cfg)

    def build(self) -> None:
        activation_name = self.critic_cfg['mlp']['activation']
        self.activation = eval(f'torch.nn.{activation_name}')
        if 'last_activation' in self.critic_cfg:
            self.last_activation = eval(f'torch.nn.{self.critic_cfg["last_activation"]}')()
        else:
            self.last_activation = lambda x: x
        self.add_module('model', MLP(
            input_size=(self.obs_dim + self.state_dim + self.stage_dim), output_size=self.reward_dim, \
            shape=self.critic_cfg['mlp']['shape'], activation=self.activation,
        ))
        for item_idx in range(len(self.critic_cfg['clip_range'])):
            item = self.critic_cfg['clip_range'][item_idx]
            if type(item) == str:
                self.critic_cfg['clip_range'][item_idx] = eval(item)
        self.clip_range = self.critic_cfg['clip_range']

    def forward(self, obs:torch.Tensor, state:torch.Tensor, stage:torch.Tensor) -> torch.Tensor:
        input_tensor = torch.cat([obs, state, stage], dim=-1)
        x = self.last_activation(self.model(input_tensor))
        x = torch.clamp(x, self.clip_range[0], self.clip_range[1])
        return x

    def getLoss(self, obs:torch.Tensor, state:torch.Tensor, stage:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.smooth_l1_loss(self.forward(obs, state, stage), target)
