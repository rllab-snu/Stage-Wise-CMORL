from algos.common.network_base import MLP, initWeights

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
import torch


class CriticBase(ABC, torch.nn.Module):
    def __init__(self, device:torch.device) -> None:
        torch.nn.Module.__init__(self)
        self.device = device

    @abstractmethod
    def getLoss(self) -> torch.Tensor:
        """
        Return action entropy given obs.
        If obs is None, use the internal obs set in the `sample` function.
        """

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize actor's parameters.
        """

class CriticS(CriticBase):
    def __init__(self, device:torch.device, obs_dim:int, critic_cfg:dict) -> None:
        super().__init__(device)

        self.obs_dim = obs_dim
        self.critic_cfg = critic_cfg
        self.build()

    def build(self) -> None:
        # for model
        activation_name = self.critic_cfg['mlp']['activation']
        self.activation = eval(f'torch.nn.{activation_name}')
        self.add_module('model', MLP(
            input_size=self.obs_dim, output_size=1, \
            shape=self.critic_cfg['mlp']['shape'], activation=self.activation,
        ))
        for item_idx in range(len(self.critic_cfg['clip_range'])):
            item = self.critic_cfg['clip_range'][item_idx]
            if type(item) == str:
                self.critic_cfg['clip_range'][item_idx] = eval(item)
        self.clip_range = self.critic_cfg['clip_range']

    def forward(self, obs:torch.Tensor) -> torch.Tensor:
        x = self.model(obs)
        x = torch.squeeze(x, dim=-1)
        x = torch.clamp(x, self.clip_range[0], self.clip_range[1])
        return x

    def getLoss(self, obs:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(self.forward(obs), target)
    
    def initialize(self) -> None:
        self.apply(initWeights)


class CriticSA(CriticBase):
    def __init__(self, device:torch.device, obs_dim:int, action_dim:int, critic_cfg:dict) -> None:
        super().__init__(device)

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.critic_cfg = critic_cfg
        self.build()

    def build(self) -> None:
        activation_name = self.critic_cfg['mlp']['activation']
        self.activation = eval(f'torch.nn.{activation_name}')
        self.add_module('model', MLP(
            input_size=self.obs_dim + self.action_dim, output_size=1, \
            shape=self.critic_cfg['mlp']['shape'], activation=self.activation,
        ))
        for item_idx in range(len(self.critic_cfg['clip_range'])):
            item = self.critic_cfg['clip_range'][item_idx]
            if type(item) == str:
                self.critic_cfg['clip_range'][item_idx] = eval(item)
        self.clip_range = self.critic_cfg['clip_range']

    def forward(self, obs:torch.Tensor, action:torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        x = self.model(x)
        x = torch.squeeze(x, dim=-1)
        x = torch.clamp(x, self.clip_range[0], self.clip_range[1])
        return x

    def getLoss(self, obs:torch.Tensor, action:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(self.forward(obs, action), target)
    
    def initialize(self) -> None:
        self.apply(initWeights)
