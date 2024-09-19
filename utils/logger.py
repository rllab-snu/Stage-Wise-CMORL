from scipy.stats import norm
import pandas as pd
import numpy as np
import pickle
import glob
import time
import os

class Logger:
    def __init__(self, log_name_list, save_dir):
        self.log_name_list = log_name_list
        self.save_dir = save_dir

        self.record_idx_dict = dict()
        self.log_dir_dict = dict()
        self.log_dict = dict()
        for log_name in self.log_name_list:
            log_dir = f'{self.save_dir}/{log_name}_log'
            if not os.path.isdir(log_dir):
                os.makedirs(log_dir)
            exist_list = glob.glob(f"{log_dir}/record_*.pkl")
            self.record_idx_dict[log_name] = len(exist_list)
            self.log_dir_dict[log_name] = log_dir
            self.log_dict[log_name] = []


    def write(self, log_name, data):
        self.log_dict[log_name].append(data)

    def writes(self, log_name, datas):
        assert type(datas) == list
        self.log_dict[log_name] += datas

    def save(self):
        for log_name in self.log_name_list:
            record_idx = self.record_idx_dict[log_name]
            log_dir = self.log_dir_dict[log_name]
            log_file_name = f"{log_dir}/record_{record_idx}.pkl"
            with open(log_file_name, 'wb') as f:
                pickle.dump(self.log_dict[log_name], f)
            self.record_idx_dict[log_name] += 1
            self.log_dict[log_name] = []

    def get_avg(self, log_name, length=1, per_episode=True):
        length = min(len(self.log_dict[log_name]), length) 
        if length == 0: return None
        if per_episode:
            temp_data = [item[1] for item in self.log_dict[log_name][-length:]]
        else:
            temp_data = [item[1]/item[0] for item in self.log_dict[log_name][-length:]]
        return np.mean(temp_data)

    def get_std(self, log_name, length=1):
        log = self.log_dict[log_name]
        length = min(len(log), length) 
        if length == 0: return None
        temp_data = [item[1] for item in log[-length:]]
        return np.std(temp_data)

    def get_square(self, log_name, length=1):
        log = self.log_dict[log_name]
        length = min(len(log), length) 
        if length == 0: return None
        temp_data = [item[1] for item in log[-length:]]
        return np.mean(np.square(temp_data))

    def get_cvar(self, log_name, cost_alpha, length=1):
        log = self.log_dict[log_name]
        length = min(len(log), length) 
        if length == 0: return None
        temp_data = [item[1] for item in log[-length:]]
        df = pd.DataFrame({"data": temp_data})
        cvar = df.quantile(q=1.0-cost_alpha)[0]
        return cvar

    def get_cvar2(self, log_name, cost_alpha, length=1):
        log = self.log_dict[log_name]
        length = min(len(log), length) 
        if length == 0: return None
        temp_data = [item[1] for item in log[-length:]]
        mean = np.mean(temp_data)
        std = np.std(temp_data)
        sigma_unit = norm.pdf(norm.ppf(cost_alpha))/cost_alpha
        cvar = mean + sigma_unit*std
        return cvar
