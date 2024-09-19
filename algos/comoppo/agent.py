from algos.common.agent_base import AgentBase
from utils import cprint

from .normalizer import ObsRMS, RewardRMS
from .storage import ReplayBuffer
from .critic import Critic
from .actor import Actor

import numpy as np
import torch
import os

EPS = 1e-8

class Agent(AgentBase):
    def __init__(self, args) -> None:
        super().__init__(
            name=args.name,
            device=args.device,
            obs_dim=args.obs_dim,
            action_dim=args.action_dim,
            save_dir=args.save_dir,
        )

        # for base
        self.stage_dim = args.num_stages
        self.state_dim = args.state_dim
        self.reward_dim = args.reward_dim
        self.cost_dim = args.cost_dim
        self.action_bound_min = args.action_bound_min
        self.action_bound_max = args.action_bound_max
        self.n_steps = args.n_steps
        self.n_envs = args.n_envs

        # for learning
        self.discount_factor = args.discount_factor
        self.critic_lr = args.critic_lr
        self.actor_lr = args.actor_lr
        self.n_critic_iters = args.n_critic_iters
        self.n_actor_iters = args.n_actor_iters
        self.max_grad_norm = args.max_grad_norm
        self.gae_coeff = args.gae_coeff

        # for CMORL
        self.preference = np.array(args.preference)
        assert len(self.preference) == self.reward_dim
        self.preference = torch.tensor(self.preference, device=self.device, dtype=torch.float32)
        self.con_coeff = args.con_coeff
        self.con_thresholds = np.array(args.con_thresholds)
        self.con_thresholds /= (1.0 - self.discount_factor)
        assert len(self.con_thresholds) == self.cost_dim

        # for normalization
        self.norm_obs = args.norm_obs
        self.norm_reward = args.norm_reward
        self.history_len = args.history_len
        self.obs_rms = ObsRMS('obs', self.obs_dim, self.history_len, self.device)
        self.state_rms = ObsRMS('state', self.state_dim, 1, self.device)
        self.reward_rms = RewardRMS('reward', self.reward_dim, self.stage_dim, self.device)

        # for PPO
        self.max_kl = args.max_kl
        self.kl_tolerance = args.kl_tolerance
        self.adaptive_lr_ratio = args.adaptive_lr_ratio
        self.adaptive_clip_ratio = args.adaptive_clip_ratio
        self.clip_ratio = args.clip_ratio

        # for symmetry
        self.is_sym_con = args.is_sym_con
        self.sym_con_threshold = args.sym_con_threshold
        self.obs_sym_mat = args.obs_sym_mat
        self.state_sym_mat = args.state_sym_mat
        self.joint_sym_mat = args.joint_sym_mat

        # declare actor and critic
        model_cfg = args.model
        self.actor = Actor(
            self.device, self.obs_dim, self.state_dim, self.stage_dim, self.action_dim, 
            self.action_bound_min, self.action_bound_max, model_cfg['actor']).to(self.device)
        self.reward_critic = Critic(
            self.device, self.obs_dim, self.state_dim, self.stage_dim, 
            self.reward_dim, model_cfg['reward_critic']).to(self.device)
        self.cost_critic = Critic(
            self.device, self.obs_dim, self.state_dim, self.stage_dim, 
            self.cost_dim, model_cfg['cost_critic']).to(self.device)

        # declare optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.reward_critic_optimizer = torch.optim.Adam(self.reward_critic.parameters(), lr=self.critic_lr)
        self.cost_critic_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=self.critic_lr)

        # declare replay buffer
        self.replay_buffer = ReplayBuffer(
            self.device, self.discount_factor, self.gae_coeff, self.n_envs, self.n_steps, 
            self.obs_dim, self.state_dim, self.stage_dim, self.action_dim, 
            self.reward_dim, self.cost_dim, self.obs_sym_mat, self.state_sym_mat)

    ################
    # Public Methods
    ################

    @torch.no_grad()
    def getAction(
        self, obs_tensor:torch.tensor, state_tensor:torch.tensor, stage_tensor:torch.tensor, deterministic:bool) -> torch.tensor:

        norm_obs_tensor = self.obs_rms.normalize(obs_tensor)
        norm_state_tensor = self.state_rms.normalize(state_tensor)
        epsilon_tensor = torch.randn(norm_obs_tensor.shape[:-1] + (self.action_dim,), device=self.device)

        self.actor.updateActionDist(norm_obs_tensor, norm_state_tensor, stage_tensor, epsilon_tensor)
        norm_action_tensor, unnorm_action_tensor = self.actor.sample(deterministic)
        log_prob_tensor = self.actor.getLogProb()

        self.obs_tensor = obs_tensor.clone()
        self.state_tensor = state_tensor.clone()
        self.stage_tensor = stage_tensor.clone()
        self.action_tensor = norm_action_tensor.clone()
        self.log_prob_tensor = log_prob_tensor.clone()
        return unnorm_action_tensor

    def step(
            self, reward_tensor, cost_tensor, done_tensor, fail_tensor, 
            next_obs_tensor, next_state_tensor, next_stage_tensor):

        self.replay_buffer.addTransition(
            self.obs_tensor, self.state_tensor, self.stage_tensor, self.action_tensor, self.log_prob_tensor,
            reward_tensor, cost_tensor, done_tensor, fail_tensor, next_obs_tensor, next_state_tensor, next_stage_tensor)

        # update statistics
        if self.norm_obs:
            self.obs_rms.update(self.obs_tensor)
            self.state_rms.update(self.state_tensor)
        if self.norm_reward:
            self.reward_rms.update(reward_tensor, self.stage_tensor)

    def readyToTrain(self):
        return True

    def train(self):
        # convert to tensor
        with torch.no_grad():
            obs_tensor, states_tensor, stages_tensor, actions_tensor, reward_targets_tensor, cost_targets_tensor, \
            reward_gaes_tensor, cost_gaes_tensor, sym_obs_tensor, sym_states_tensor = \
                self.replay_buffer.getBatches(
                    self.obs_rms, self.state_rms, self.reward_rms, self.actor, self.reward_critic, self.cost_critic)

        # ======================= update critic ======================= #
        for _ in range(self.n_critic_iters):
            # for reward
            reward_critic_loss = self.reward_critic.getLoss(obs_tensor, states_tensor, stages_tensor, reward_targets_tensor)
            self.reward_critic_optimizer.zero_grad()
            reward_critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.reward_critic.parameters(), self.max_grad_norm)
            self.reward_critic_optimizer.step()

            # for cost
            cost_critic_loss = self.cost_critic.getLoss(obs_tensor, states_tensor, stages_tensor, cost_targets_tensor)
            self.cost_critic_optimizer.zero_grad()
            cost_critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.cost_critic.parameters(), self.max_grad_norm)
            self.cost_critic_optimizer.step()
        # ============================================================= #

        # ======================== update actor ======================== #
        with torch.no_grad():
            # calcuate actions
            epsilons_tensor = torch.randn_like(actions_tensor)
            self.actor.updateActionDist(obs_tensor, states_tensor, stages_tensor, epsilons_tensor)
            old_action_dists = self.actor.getDist()
            old_action_means = self.actor.getMeanStd()[0]
            old_log_probs = self.actor.getLogProb(actions_tensor)

            # calculate constraints & symmetric actions
            sym_old_action_means = torch.matmul(old_action_means, self.joint_sym_mat)
            con_vals = cost_targets_tensor.mean(dim=0)

            # calculate reduced GAEs
            reward_gaes_tensor -= reward_gaes_tensor.mean(dim=0, keepdim=True)
            reduced_gaes_tensor = (reward_gaes_tensor*self.preference.unsqueeze(0)).sum(dim=-1)
            reduced_gaes_tensor /= (reduced_gaes_tensor.std() + EPS)
            cost_gaes_tensor = (cost_gaes_tensor - cost_gaes_tensor.mean(dim=0, keepdim=True)) \
                                /(cost_gaes_tensor.std(dim=0, keepdim=True) + EPS)
            for cost_idx in range(self.cost_dim):
                if con_vals[cost_idx] > self.con_thresholds[cost_idx]:
                    reduced_gaes_tensor -= self.con_coeff*cost_gaes_tensor[:, cost_idx]
            reduced_gaes_tensor /= (reduced_gaes_tensor.std() + EPS)

        for _ in range(self.n_actor_iters):
            self.actor.updateActionDist(obs_tensor, states_tensor, stages_tensor, epsilons_tensor)
            cur_action_dists = self.actor.getDist()
            kl = torch.mean(torch.sum(torch.distributions.kl.kl_divergence(old_action_dists, cur_action_dists), dim=-1))
            if kl > self.max_kl*self.kl_tolerance: break
            cur_log_probs = self.actor.getLogProb(actions_tensor)
            prob_ratios = torch.exp(cur_log_probs - old_log_probs)
            clipped_ratios = torch.clamp(prob_ratios, min=1.0-self.clip_ratio, max=1.0+self.clip_ratio)
            actor_loss = -torch.mean(torch.minimum(reduced_gaes_tensor*prob_ratios, reduced_gaes_tensor*clipped_ratios))
            # ========= symmetry constraint ========= #
            sym_action_means = self.actor(sym_obs_tensor, sym_states_tensor, stages_tensor)[0]
            sym_constraints = torch.abs(sym_action_means - sym_old_action_means)
            sym_constraint = sym_constraints.mean()
            if self.is_sym_con and sym_constraint > self.sym_con_threshold:
                actor_loss += self.con_coeff*sym_constraint
            # ======================================= #
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

        self.actor.updateActionDist(obs_tensor, states_tensor, stages_tensor, epsilons_tensor)
        cur_action_dists = self.actor.getDist()
        kl = torch.mean(torch.sum(torch.distributions.kl.kl_divergence(old_action_dists, cur_action_dists), dim=-1))
        entropy = self.actor.getEntropy()

        # adjust learning rate based on KL divergence
        if kl > self.max_kl*self.kl_tolerance:
            self.clip_ratio /= self.adaptive_clip_ratio
            self.actor_lr /= self.adaptive_lr_ratio
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = self.actor_lr
        elif kl < self.max_kl/self.kl_tolerance:
            self.clip_ratio *= self.adaptive_clip_ratio
            self.actor_lr *= self.adaptive_lr_ratio
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = self.actor_lr
        self.clip_ratio = np.clip(self.clip_ratio, 0.0, 0.2)
        # ============================================================== #

        # upgrade rms
        if self.norm_obs:
            self.obs_rms.upgrade()
            self.state_rms.upgrade()

        # return
        train_results = {
            'objectives': reward_targets_tensor.mean(dim=0).detach().cpu().numpy(),
            'constraints': con_vals.detach().cpu().numpy(),
            'reward_critic_loss': reward_critic_loss.item(),
            'cost_critic_loss': cost_critic_loss.item(),
            'entropy': entropy.item(),
            'kl': kl.item(),
            'actor_lr': self.actor_lr,
            'clip_ratio': self.clip_ratio,
            'sym_constraint': sym_constraint.item(),
        }
        for stage_idx in range(self.stage_dim):
            train_results[f'stage_{stage_idx}'] = stages_tensor[:, stage_idx].mean().item()
        return train_results
    
    def save(self, model_num):
        # save rms
        self.obs_rms.save(self.save_dir, model_num)
        self.state_rms.save(self.save_dir, model_num)
        self.reward_rms.save(self.save_dir, model_num)

        # save network models
        save_dict = {
            'actor': self.actor.state_dict(),
            'reward_critic': self.reward_critic.state_dict(),
            'cost_critic': self.cost_critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'reward_critic_optimizer': self.reward_critic_optimizer.state_dict(),
            'cost_critic_optimizer': self.cost_critic_optimizer.state_dict(),
        }
        torch.save(save_dict, f"{self.save_dir}/checkpoint/model_{model_num}.pt")
        cprint(f'[{self.name}] save success.', bold=True, color="blue")

    def load(self, model_num):
        # load rms
        self.obs_rms.load(self.save_dir, model_num)
        self.state_rms.load(self.save_dir, model_num)
        self.reward_rms.load(self.save_dir, model_num)

        # load network models
        if not os.path.isdir(f"{self.save_dir}/checkpoint"):
            os.makedirs(f"{self.save_dir}/checkpoint")
        checkpoint_file = f"{self.save_dir}/checkpoint/model_{model_num}.pt"
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor'])
            self.reward_critic.load_state_dict(checkpoint['reward_critic'])
            self.cost_critic.load_state_dict(checkpoint['cost_critic'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.reward_critic_optimizer.load_state_dict(checkpoint['reward_critic_optimizer'])
            self.cost_critic_optimizer.load_state_dict(checkpoint['cost_critic_optimizer'])
            cprint(f'[{self.name}] load success.', bold=True, color="blue")
            return int(model_num)
        else:
            self.actor.initialize()
            self.reward_critic.initialize()
            self.cost_critic.initialize()
            cprint(f'[{self.name}] load fail.', bold=True, color="red")
            return 0

    ################
    # private method
    ################
