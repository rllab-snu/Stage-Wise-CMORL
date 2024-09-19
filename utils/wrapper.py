import numpy as np
import pickle

EPS = 1e-8

class EnvWrapper:
    def __init__(self, env_fn) -> None:
        self._env = env_fn()
        self.device = self._env.rl_device
        self.n_envs = self._env.num_envs
        self.obs_dim = self._env.num_obs
        self.state_dim = self._env.num_states

    #################
    # public function
    #################

    def reset(self, **kwargs):
        obs_dict = self._env.reset(**kwargs)
        obs = obs_dict['obs']
        state = obs_dict['states']
        return obs, state
    
    def step(self, action):
        obs_dict, reward, done, info = self._env.step(action)
        obs = obs_dict['obs']
        state = obs_dict['states']
        return obs, state, reward, done, info
    
    def close(self):
        del self._env
    
    @property
    def unwrapped(self):
        return self._env
