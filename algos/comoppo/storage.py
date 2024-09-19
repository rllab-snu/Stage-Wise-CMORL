from copy import deepcopy
import numpy as np
import random
import torch
import os

EPS = 1e-8 

class ReplayBuffer:
    def __init__(
            self, device:torch.device, 
            discount_factor:float, 
            gae_coeff:float, 
            n_envs:int,
            n_steps:int, 
            obs_dim:int,
            state_dim:int,
            stage_dim:int,
            action_dim:int,
            reward_dim:int,
            cost_dim:int,
            obs_sym_mat:torch.Tensor,
            state_sym_mat:torch.Tensor) -> None:

        self.device = device
        self.discount_factor = discount_factor
        self.gae_coeff = gae_coeff
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.stage_dim = stage_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.cost_dim = cost_dim
        self.obs_sym_mat = obs_sym_mat
        self.state_sym_mat = state_sym_mat

        self.n_steps_per_env = int(self.n_steps/self.n_envs)

        self.cnt = 0
        self.obs = torch.zeros((self.n_steps_per_env, self.n_envs, self.obs_dim), device=self.device, dtype=torch.float32)
        self.states = torch.zeros((self.n_steps_per_env, self.n_envs, self.state_dim), device=self.device, dtype=torch.float32)
        self.stages = torch.zeros((self.n_steps_per_env, self.n_envs, self.stage_dim), device=self.device, dtype=torch.float32)
        self.actions = torch.zeros((self.n_steps_per_env, self.n_envs, self.action_dim), device=self.device, dtype=torch.float32)
        self.log_probs = torch.zeros((self.n_steps_per_env, self.n_envs), device=self.device, dtype=torch.float32)
        self.rewards = torch.zeros((self.n_steps_per_env, self.n_envs, self.reward_dim), device=self.device, dtype=torch.float32)
        self.costs = torch.zeros((self.n_steps_per_env, self.n_envs, self.cost_dim), device=self.device, dtype=torch.float32)
        self.dones = torch.zeros((self.n_steps_per_env, self.n_envs), device=self.device, dtype=torch.float32)
        self.fails = torch.zeros((self.n_steps_per_env, self.n_envs), device=self.device, dtype=torch.float32)
        self.next_obs = torch.zeros((self.n_steps_per_env, self.n_envs, self.obs_dim), device=self.device, dtype=torch.float32)
        self.next_states = torch.zeros((self.n_steps_per_env, self.n_envs, self.state_dim), device=self.device, dtype=torch.float32)
        self.next_stages = torch.zeros((self.n_steps_per_env, self.n_envs, self.stage_dim), device=self.device, dtype=torch.float32)

    ################
    # Public Methods
    ################

    def getLen(self):
        return self.n_envs*self.cnt

    def addTransition(
            self, obs, states, stages, actions, log_probs, rewards, costs, dones, fails, 
            next_obs, next_states, next_stages):
        
        self.obs[self.cnt, :, :] = obs
        self.states[self.cnt, :, :] = states
        self.stages[self.cnt, :, :] = stages
        self.actions[self.cnt, :, :] = actions
        self.log_probs[self.cnt, :] = log_probs
        self.rewards[self.cnt, :, :] = rewards
        self.costs[self.cnt, :, :] = costs
        self.dones[self.cnt, :] = dones
        self.fails[self.cnt, :] = fails
        self.next_obs[self.cnt, :, :] = next_obs
        self.next_states[self.cnt, :, :] = next_states
        self.next_stages[self.cnt, :, :] = next_stages
        self.cnt += 1
        if self.cnt == self.n_steps_per_env:
            self.cnt = 0

    @torch.no_grad()
    def getBatches(self, obs_rms, state_rms, reward_rms, actor, reward_critic, cost_critic):
        # process the latest trajectories
        assert self.cnt == 0
        end_idx = self.n_steps_per_env
        start_idx = end_idx - self.n_steps_per_env
        obs_tensor, states_tensor, stages_tensor, actions_tensor, reward_targets_tensor, cost_targets_tensor, \
        reward_gaes_tensor, cost_gaes_tensor, sym_obs_tensor, sym_states_tensor = \
            self._processBatches(obs_rms, state_rms, reward_rms, actor, reward_critic, cost_critic, start_idx, end_idx)
        return obs_tensor, states_tensor, stages_tensor, actions_tensor, reward_targets_tensor, cost_targets_tensor, \
            reward_gaes_tensor, cost_gaes_tensor, sym_obs_tensor, sym_states_tensor
        
    #################
    # Private Methods
    #################

    def _processBatches(self, obs_rms, state_rms, reward_rms, actor, reward_critic, cost_critic, start_idx, end_idx):
        obs_tensor = self.obs[start_idx:end_idx, :, :]
        states_tensor = self.states[start_idx:end_idx, :, :]
        stages_tensor = self.stages[start_idx:end_idx, :, :]
        actions_tensor = self.actions[start_idx:end_idx, :, :]
        mu_log_probs_tensor = self.log_probs[start_idx:end_idx, :]
        rewards_tensor = self.rewards[start_idx:end_idx, :, :]
        costs_tensor = self.costs[start_idx:end_idx, :, :]
        dones_tensor = self.dones[start_idx:end_idx, :]
        fails_tensor = self.fails[start_idx:end_idx, :]
        next_obs_tensor = self.next_obs[start_idx:end_idx, :, :]
        next_states_tensor = self.next_states[start_idx:end_idx, :, :]
        next_stages_tensor = self.next_stages[start_idx:end_idx, :, :]

        # get stages
        stage_completes_tensor = (1.0 - (stages_tensor*next_stages_tensor).sum(dim=-1, keepdim=True)) # (n_steps_per_env, n_envs, 1)
        stage_completes_tensor = (stage_completes_tensor > 0.5).type(torch.float32)

        # normalize 
        norm_obs_tensor = obs_rms.normalize(obs_tensor)
        norm_states_tensor = state_rms.normalize(states_tensor)
        norm_next_obs_tensor = obs_rms.normalize(next_obs_tensor)
        norm_next_states_tensor = state_rms.normalize(next_states_tensor)
        norm_reward_std = 0.1
        norm_reward_mean = 0.1
        norm_rewards_tensor = reward_rms.normalize(
            rewards_tensor, stages_tensor, default_mean=norm_reward_mean, default_std=norm_reward_std)
        norm_rewards_tensor += norm_rewards_tensor*stage_completes_tensor*self.discount_factor/(1.0 - self.discount_factor)
        norm_costs_tensor = fails_tensor.unsqueeze(-1)*costs_tensor/(1.0 - self.discount_factor) \
                        + (1.0 - fails_tensor.unsqueeze(-1))*costs_tensor

        # symmetrize
        sym_obs_tensor = obs_rms.normalize(torch.matmul(obs_tensor, self.obs_sym_mat))
        sym_states_tensor = state_rms.normalize(torch.matmul(states_tensor, self.state_sym_mat))
        
        # get values
        next_reward_values_tensor = reward_critic(norm_next_obs_tensor, norm_next_states_tensor, next_stages_tensor) # (n_steps_per_env, n_envs, reward_dim)
        reward_values_tensor = reward_critic(norm_obs_tensor, norm_states_tensor, stages_tensor) # (n_steps_per_env, n_envs, reward_dim)
        next_cost_values_tensor = cost_critic(norm_next_obs_tensor, norm_next_states_tensor, next_stages_tensor) # (n_steps_per_env, n_envs, cost_dim)
        cost_values_tensor = cost_critic(norm_obs_tensor, norm_states_tensor, stages_tensor) # (n_steps_per_env, n_envs, cost_dim)

        # get targets
        reward_delta = torch.zeros_like(rewards_tensor[0]) # (n_envs, reward_dim)
        reward_targets = torch.zeros_like(rewards_tensor) # (n_steps_per_env, n_envs, reward_dim)
        cost_delta = torch.zeros_like(costs_tensor[0]) # (n_envs, cost_dim)
        cost_targets = torch.zeros_like(costs_tensor) # (n_steps_per_env, n_envs, cost_dim)
        for t in reversed(range(len(reward_targets))):
            reward_targets[t, :, :] = norm_rewards_tensor[t, :, :] \
                                    + self.discount_factor*(1.0 - fails_tensor[t, :].unsqueeze(-1))*next_reward_values_tensor[t, :, :] \
                                    + self.discount_factor*(1.0 - dones_tensor[t, :].unsqueeze(-1))*reward_delta
            reward_delta = self.gae_coeff*(reward_targets[t] - reward_values_tensor[t])
            cost_targets[t, :, :] = norm_costs_tensor[t, :, :] \
                                + self.discount_factor*(1.0 - fails_tensor[t, :].unsqueeze(-1))*next_cost_values_tensor[t, :, :] \
                                + self.discount_factor*(1.0 - dones_tensor[t, :].unsqueeze(-1))*cost_delta
            cost_delta = self.gae_coeff*(cost_targets[t] - cost_values_tensor[t])
        reward_gaes = reward_targets - reward_values_tensor
        cost_gaes = cost_targets - cost_values_tensor
                    
        return norm_obs_tensor.view(-1, self.obs_dim), norm_states_tensor.view(-1, self.state_dim), stages_tensor.view(-1, self.stage_dim), \
                actions_tensor.view(-1, self.action_dim), reward_targets.view(-1, self.reward_dim), cost_targets.view(-1, self.cost_dim), \
                reward_gaes.view(-1, self.reward_dim), cost_gaes.view(-1, self.cost_dim), \
                sym_obs_tensor.view(-1, self.obs_dim), sym_states_tensor.view(-1, self.state_dim)