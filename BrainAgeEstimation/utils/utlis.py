import argparse
import json
import os
import random

import numpy as np
import torch

from torch import Tensor
from typing import Optional


def seed_everything(seed: int = -1):
    """
    Set seed for everything (random, numpy, cuda...)
    :param seed: int
    :return: None
    """
    if seed == -1:
        seed = np.random.randint(0, 100000000)
    print(f"seed for seed_everything(): {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)  # set random seed for numpy
    torch.manual_seed(seed)  # set random seed for CPU
    torch.cuda.manual_seed_all(seed)  # set random seed for all GPUs


def normalize_data(x, norm_mode='standard'):
    num_Patient, num_Feature = np.shape(x)
    if norm_mode is None:
        return x
    if norm_mode == 'standard':  # zero mean unit variance
        for j in range(num_Feature):
            if np.std(x[:, j]) != 0:
                x[:, j] = (x[:, j] - np.mean(x[:, j])) / np.std(x[:, j])
            else:
                x[:, j] = (x[:, j] - np.mean(x[:, j]))
    elif norm_mode == 'normal':  # min-max normalization
        for j in range(num_Feature):
            x[:, j] = (x[:, j] - np.min(x[:, j])) / (np.max(x[:, j]) - np.min(x[:, j]))
    else:
        print("INPUT MODE ERROR!")

    return x


def mkdir_if_needed(folder: str):
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)


def to_device(device, *arrays):
    if len(arrays) == 0:
        return None
    result = [array.to(device) for array in arrays]
    return tuple(result)


def read_img(ids, image_path):
    if ids is None or len(ids) == 0:
        print('No image needs to read.')
        return None
    img_list = []
    for idx in ids:
        img = np.load(os.path.join(image_path, f'{idx}_20227_2_0.npy'))
        img_list.append(img)
    images = np.stack(img_list, axis=0)
    return images


def save_model(state_dict, save_path: str, save_name: str = None, post_fix: str = None):
    mkdir_if_needed(save_path)
    if save_name is None:
        save_name = 'model_params'
    if post_fix is not None:
        save_name = save_name + f'_{post_fix}'
    torch.save(state_dict, os.path.join(save_path, f'{save_name}.pth'))


def save_args(args, save_path):
    with open(os.path.join(save_path, 'args.txt'), 'w') as fp:
        json.dump(args.__dict__, fp, indent=2)


def load_args(args_path):
    with open(args_path, 'r') as fp:
        args = json.load(fp)
    args = argparse.Namespace(**args)
    return args
