from algos.common.network_base import MLP, initWeights
from algos.common.actor_base import (
    ActorBase, normalize, unnormalize, clip
)

from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform
from typing import Tuple
import numpy as np
import torch

EPS = 1e-8


class ActorSquash(ActorBase):
    def __init__(self, device:torch.device, obs_dim:int, action_dim:int, \
                action_bound_min:np.ndarray, action_bound_max:np.ndarray, actor_cfg:dict, \
                log_std_min:float=-4.0, log_std_max:float=2.0) -> None:
        ActorBase.__init__(self, device)

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # for model
        self.actor_cfg = actor_cfg
        self.use_action_bound = self.actor_cfg['use_action_bound']
        if 'last_activation' in self.actor_cfg:
            self.last_activation = eval(f'torch.nn.{self.actor_cfg["last_activation"]}')()
        else:
            self.last_activation = lambda x: x
        self.activation = eval(f'torch.nn.{self.actor_cfg["mlp"]["activation"]}')
        self.log_std_init = self.actor_cfg['log_std_init']
        self.log_std_fix = self.actor_cfg['log_std_fix']

        # for action
        if self.use_action_bound:
            self.action_bound_min = torch.tensor(
                action_bound_min, device=device, dtype=torch.float32)
            self.action_bound_max = torch.tensor(
                action_bound_max, device=device, dtype=torch.float32)
        else:
            self.action_bound_min = torch.tensor(
                -np.ones(self.action_dim), device=device, dtype=torch.float32)
            self.action_bound_max = torch.tensor(
                np.ones(self.action_dim), device=device, dtype=torch.float32)

        # build model
        self.build()


    def build(self) -> None:
        self.add_module('model', MLP(
            input_size=self.obs_dim, output_size=self.actor_cfg['mlp']['shape'][-1], \
            shape=self.actor_cfg['mlp']['shape'][:-1], activation=self.activation,
        ))
        self.add_module("mean_decoder", torch.nn.Sequential(
            self.activation(),
            torch.nn.Linear(self.actor_cfg['mlp']['shape'][-1], self.action_dim),
        ))
        if self.log_std_fix:
            self.std_decoder = lambda x: torch.ones(
                *x.shape[:-1], self.action_dim, dtype=torch.float, device=self.device)*self.log_std_init
        else:
            self.add_module("std_decoder", torch.nn.Sequential(
                self.activation(),
                torch.nn.Linear(self.actor_cfg['mlp']['shape'][-1], self.action_dim),
            ))
        
    def forward(self, state:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        output: (mean, log_std, std)
        '''
        x = self.model(state)
        mean = self.last_activation(self.mean_decoder(x))
        log_std = self.std_decoder(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mean, log_std, std

    def updateActionDist(self, state:torch.Tensor, epsilon:torch.Tensor) -> None:
        self.action_mean, self.action_log_std, self.action_std = self.forward(state)
        self.action_dist = torch.distributions.Normal(self.action_mean, self.action_std)
        self.action_dist = TransformedDistribution(self.action_dist, TanhTransform())
        self.normal_action = self.action_mean + epsilon*self.action_std

    def sample(self, deterministic:bool=False) -> Tuple[torch.Tensor, torch.Tensor]:
        if deterministic:
            action = self.action_mean
        else:
            action = self.normal_action
        norm_action = torch.tanh(action)
        unnorm_action = unnormalize(norm_action, self.action_bound_min, self.action_bound_max)
        return norm_action, unnorm_action

    def getNormalAction(self) -> torch.Tensor:
        return self.normal_action
    
    def getDist(self) -> torch.distributions.Distribution:
        return self.action_dist
        
    def getEntropy(self) -> torch.Tensor:
        '''
        return entropy of action distribution given state.
        '''
        log_prob = self.getLogProbJIT(self.normal_action, self.action_mean, self.action_std)
        entropy = -torch.mean(log_prob)
        return entropy
    
    def getLogProb(self, action=None) -> torch.Tensor:
        '''
        return log probability of action given state.
        '''
        if action is None:
            return self.getLogProbJIT(self.normal_action, self.action_mean, self.action_std)
        else:
            return self.getLogProbJIT(action, self.action_mean, self.action_std)

    @torch.jit.export
    def getLogProbJIT(self, normal_action, mean, std):
        normal = torch.distributions.Normal(mean, std)
        log_prob = torch.sum(normal.log_prob(normal_action), dim=-1)
        log_prob -= torch.sum(2.0*(np.log(2.0) - normal_action - torch.nn.functional.softplus(-2.0*normal_action)), dim=-1)
        return log_prob

    def initialize(self) -> None:
        for name, module in self.named_children():
            if name == 'std_decoder':
                initializer = lambda m: initWeights(m, init_bias=self.log_std_init)
            else:
                initializer = lambda m: initWeights(m)
            module.apply(initializer)
