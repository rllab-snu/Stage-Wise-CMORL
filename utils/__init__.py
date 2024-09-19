from shutil import copyfile
import numpy as np
import random
import ntpath
import torch
import os

def backupFiles(log_dir, task_save_items=None, algo_save_items=None):
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    if task_save_items is not None:
        task_log_dir = f"{log_dir}/task"
        if not os.path.isdir(task_log_dir):
            os.makedirs(task_log_dir)
        for save_item in task_save_items:
            base_file_name = ntpath.basename(save_item)
            copyfile(save_item, f"{task_log_dir}/{base_file_name}")
    if algo_save_items is not None:
        algo_log_dir = f"{log_dir}/algo"
        if not os.path.isdir(algo_log_dir):
            os.makedirs(algo_log_dir)
        for save_item in algo_save_items:
            base_file_name = ntpath.basename(save_item)
            copyfile(save_item, f"{algo_log_dir}/{base_file_name}")

def setSeed(seed):
    # for random seed
    np.random.seed(seed)
    random.seed(seed)    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def cprint(str, bold=False, underline=False, color=''):
    color_encoding = ''
    if color == 'blue':
        color_encoding = bcolors.OKBLUE
    elif color == 'cyan':
        color_encoding = bcolors.OKCYAN
    elif color == "orange":
        color_encoding = bcolors.WARNING
    elif color == 'red':
        color_encoding = bcolors.FAIL
    bold_encoding = bcolors.BOLD if bold else ''
    underline_encoding = bcolors.UNDERLINE if underline else ''
    print(bold_encoding + underline_encoding + color_encoding + str + bcolors.ENDC)
