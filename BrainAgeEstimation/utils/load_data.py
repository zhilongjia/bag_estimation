import os
from typing import Optional
from utils.utlis import normalize_data
import pandas as pd
import numpy as np


def load_data_ukb(data_path, normalize=True, norm_mode='standard', train: bool = True, sex: int = -1):
    if train:
        df = pd.read_csv(os.path.join(data_path, 'csv', 'ukb_data_healthy_v2.csv'))
        df = df[df['usable'] == 1]
        # select healthy patients for training
        df = df[df['train'] == 1]
    else:
        # select patients for testing
        df = pd.read_csv(os.path.join(data_path, 'csv', 'ukb_data_healthy_v2.csv'))
        df = df[df['usable'] == 1]
        df = df[df['train'] == 0]
        # select healthy patients for training
        # df = df[df['is_healthy'] == 0]
        # pass
    df['train'] = 1

    if sex != -1:
        # sex : -1: all, 0: female, 1: male.
        df = df[df['sex'] == sex]

    img = np.asarray(df['Eid'])
    x = np.asarray(df.iloc[:, 10:])
    y = np.asarray(df[['age']])

    if normalize:
        x = normalize_data(x, norm_mode=norm_mode)

    return img, x, y


def load_external_data(dataset, data_path):
    # df should contain at least 2 columns for the data:
    # 1: image_id (image data should be named by image_id.nii.gz or image_id.npy)
    #
    # 2: (chronological) age when took the MR image

    df = pd.read_csv(os.path.join(data_path, 'csv', f'{dataset}_data.csv'))
    if dataset == 'ixi':
        df = df[(df['age'] > 45) & (df['age'] < 85)]
    img = np.asarray(df['img_id'])
    y = np.asarray(df[['age']])

    return img, y

