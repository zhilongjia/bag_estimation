import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils.load_data import load_data_ukb, load_external_data
from utils.datasets import TestDataset, get_transform

dataset = 'ADNI'

cwd = os.getcwd()
data_path = os.path.join(cwd, 'ext', dataset)
image_path = os.path.join(data_path, 'img_data')
save_path = os.path.join(data_path, 'npy_data')

if not os.path.exists(save_path):
    os.mkdir(save_path)

# load data
img, y = load_external_data(dataset.lower(), data_path)
print(len(img))

train_set = TestDataset(img, y, image_path=image_path, transform=get_transform(), image_suffix=None)
train_loader = DataLoader(train_set, batch_size=1, shuffle=False)

# pre transform data from nii to npy
for i, data in enumerate(train_loader):
    img, y, eid = data
    img = img.detach().cpu().numpy()
    img = img.reshape(1, 128, 128, 128)
    if isinstance(eid, torch.Tensor):
        eid = eid.item()
    else:
        eid = eid[0]
    print(i)
    if not os.path.exists(os.path.join(save_path, f'{eid}.npy')):
        np.save(os.path.join(save_path, f'{eid}.npy'), img)
    else:
        print('saved.')




