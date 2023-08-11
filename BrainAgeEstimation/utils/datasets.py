import os

import torch
from torch.utils.data import Dataset
import numpy as np
from monai import transforms as mo_trans


class PretrainDataset(Dataset):
    def __init__(self, img, x, image_path, img_type='nii', transform=None):
        self.img = img
        self.x = torch.from_numpy(x).float()
        self.len = x.shape[0]
        self.transform = transform
        self.image_path = image_path
        self.image_type = img_type

    def __getitem__(self, index):
        # img_data = read_img(self.img[index], self.image_path)
        if self.image_type == 'npy':
            img_path = os.path.join(self.image_path, f'{self.img[index]}_20252_2_0.npy')
            img_data = np.load(img_path, allow_pickle=True)
            img_data = img_data.astype(np.float32)
        else:
            img_data = os.path.join(self.image_path, f'{self.img[index]}_20252_2_0.nii.gz')

        if self.transform is not None:
            img_data = self.transform(img_data)

        # img_data = torch.from_numpy(img_data).float()
        return img_data, self.x[index]

    def __len__(self):
        return self.len


# for training model
class NewDataset(Dataset):
    def __init__(self, img, y, image_path, img_type='nii', transform=None, image_suffix='_20252_2_0'):
        self.img = img
        self.y = torch.from_numpy(y).float()
        self.len = y.shape[0]
        self.transform = transform
        self.image_path = image_path
        self.image_type = img_type
        self.image_suffix = image_suffix if image_suffix is not None else ''

    def __getitem__(self, index):
        if self.image_type == 'npy':
            img_path = os.path.join(self.image_path, f'{self.img[index]}{self.image_suffix}.npy')
            img_data = np.load(img_path, allow_pickle=True)
            img_data = torch.from_numpy(img_data).float()
        else:
            img_data = os.path.join(self.image_path, f'{self.img[index]}{self.image_suffix}.nii.gz')

        if self.transform is not None:
            img_data = self.transform(img_data)

        return img_data, self.y[index]

    def __len__(self):
        return self.len


# for testing
class TestDataset(Dataset):
    def __init__(self, img, y, image_path, img_type='nii', transform=None, image_suffix='_20252_0'):
        self.img = img
        self.y = torch.from_numpy(y).float()
        self.len = y.shape[0]
        self.transform = transform
        self.image_path = image_path
        self.image_type = img_type
        self.image_suffix = image_suffix if image_suffix is not None else ''

    def __getitem__(self, index):
        if self.image_type == 'npy':
            img_path = os.path.join(self.image_path, f'{self.img[index]}{self.image_suffix}.npy')
            img_data = np.load(img_path, allow_pickle=True)
            img_data = torch.from_numpy(img_data).float()
        else:
            img_data = os.path.join(self.image_path, f'{self.img[index]}{self.image_suffix}.nii.gz')

        if self.transform is not None:
            img_data = self.transform(img_data)

        return img_data, self.y[index], self.img[index]

    def __len__(self):
        return self.len


# for model interpretation
class ImgDataset(Dataset):
    def __init__(self, img, image_path, img_type='nii', transform=None):
        self.img = img
        self.len = len(img)
        self.transform = transform
        self.image_path = image_path
        self.image_type = img_type

    def __getitem__(self, index):
        if self.image_type == 'npy':
            img_path = os.path.join(self.image_path, f'{self.img[index]}_20252_2_0.npy')
            img_data = np.load(img_path, allow_pickle=True)
            img_data = torch.from_numpy(img_data).float()
        else:
            img_data = os.path.join(self.image_path, f'{self.img[index]}_20252_2_0.nii.gz')

        if self.transform is not None:
            img_data = self.transform(img_data)

        return img_data, self.img[index]

    def __len__(self):
        return self.len


class MaskedDataset(Dataset):
    def __init__(self, img, y, image_path, img_type='nii', image_suffix='_20252_2_0', mask=None):
        self.img = img
        self.y = torch.from_numpy(y).float()
        self.len = y.shape[0]
        self.image_path = image_path
        self.image_type = img_type
        self.image_suffix = image_suffix if image_suffix is not None else ''

        self.mask = mask

        # read nii.gz images -> size: [182, 218, 182, 1]
        self.transform1 = mo_trans.Compose([
            mo_trans.LoadImage(image_only=True, ensure_channel_first=True)
        ])
        # transform to [128, 128, 128, 1]
        self.transform2 = mo_trans.Compose([
            mo_trans.CenterSpatialCrop(roi_size=(182, 182, 182)),
            mo_trans.Resize(spatial_size=(128, 128, 128)),
            mo_trans.NormalizeIntensity(nonzero=True),
            mo_trans.ToTensor()
        ])

    def __getitem__(self, index):
        if self.image_type == 'npy':
            img_path = os.path.join(self.image_path, f'{self.img[index]}{self.image_suffix}.npy')
            img_data = np.load(img_path, allow_pickle=True)
            img_data = torch.from_numpy(img_data).float()
        else:
            img_data = os.path.join(self.image_path, f'{self.img[index]}{self.image_suffix}.nii.gz')

        img_data = self.transform1(img_data)
        if self.mask is not None:
            img_data = img_data * self.mask
        img_data = self.transform2(img_data)
        return img_data, self.y[index], self.img[index]

    def __len__(self):
        return self.len


def get_transform(image_type='nii', data_type='3d', norm='std', to_tensor=True):
    t = []
    if image_type == 'npy':
        t.append(mo_trans.AddChannel())
    else:
        t.append(mo_trans.LoadImage(image_only=True, ensure_channel_first=True))

    if data_type == '3d':
        t.append(mo_trans.CenterSpatialCrop(roi_size=(182, 182, 182)))
        t.append(mo_trans.Resize(spatial_size=(128, 128, 128)))
    else:
        t.append(mo_trans.CenterSpatialCrop(roi_size=(182, 182)))



    if norm == 'maxmin':
        pass
    elif norm == 'std':
        t.append(mo_trans.NormalizeIntensity(nonzero=True))  # std mean norm

    if to_tensor:
        t.append(mo_trans.ToTensor())
    else:
        t.append(mo_trans.ToNumpy())

    return mo_trans.Compose(t)
