import argparse

import torch
from torch.autograd import Variable
from torch.nn import functional as F
import os
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm

from model.brainAgeEstimator2 import BrainAgeEstimator
from utils.datasets import TestDataset, get_transform, NewDataset
from utils.load_data import load_data_ukb
from utils.utlis import seed_everything, mkdir_if_needed
from utils.utlis import to_device


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_saliency_maps(model, test_set):
    model.eval()  # 将模型设置为评估模式
    saliency_maps = []

    # set progress bar
    pbar = tqdm(enumerate(test_set), total=len(test_set), desc='compute saliency maps')
    # iterate over test data with progress bar
    for i, data in pbar:
        (image, y) = data
        image, y = to_device(device, image, y)
        image = image.requires_grad_()  # 设置图像需要梯度

        # get prediction and calculate loss
        prediction = model.vitonly(image)

        # MSE loss
        loss = F.mse_loss(prediction, y)

        # backpropagation to calculate gradients
        model.zero_grad()
        loss.backward()

        # get gradients for input image
        saliency = image.grad.data.abs()

        # max across channels
        saliency, _ = torch.max(saliency, dim=1)

        # saliency map to numpy array
        saliency = saliency.detach().cpu().numpy()

        # image to numpy
        image = image.detach().cpu().numpy().reshape(1, 128, 128, 128)

        # set saliency of negative and zero values in image to 0
        saliency[image <= 0] = 0

        # min-max normalize
        saliency = (saliency - np.min(saliency)) / (np.max(saliency) - np.min(saliency))

        # append saliency map to list
        saliency_maps.append(saliency)

        # add loss to progress bar
        pbar.set_postfix(**{'loss MSE=': loss.item()})

    saliency_maps = np.concatenate(saliency_maps)  # 堆叠所有显著性图形成一个批次
    return saliency_maps


def compute_saliency_probability_maps(saliency_maps):
    # saliency_maps: ndarray -> [b d h w]
    # sum of the saliency map for each image
    saliency_sum = saliency_maps.reshape(saliency_maps.shape[0], -1).sum(1)

    # convert to probability map
    saliency_probability_maps = saliency_maps / saliency_sum[:, None, None, None]

    return saliency_probability_maps


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(args.seed)

    cwd = os.getcwd()

    data_path = os.path.join(cwd, 'data')
    img_type = args.img_type

    if img_type == 'nii':
        image_path = os.path.join(data_path, 't1_data')
    else:
        image_path = os.path.join(data_path, 'npy_data')

    model_name = args.model_name
    res_path = os.path.join(cwd, 'results', model_name)
    mkdir_if_needed(res_path)

    # load test data
    img, x, y = load_data_ukb(data_path, train=False, healthy=args.is_healthy)
    print(args.is_healthy)
    print(f'{len(img)} for testing.')

    # use 5 samples for testing code
    # img = img[0:5]
    # x = x[0:5]
    # y = y[0:5]

    x_dim = x.shape[1]
    channels, depth, img_size = 1, 128, (128, 128)
    image_patch_size, depth_patch_size = 16, 16

    fold = args.fold
    assert fold in [1, 2, 3, 4, 5], 'fold must be in [1, 2, 3, 4, 5]'

    trained_weight = os.path.join(cwd, 'saved_models', model_name, f'model_vit_fold{fold}.pth')

    test_set = NewDataset(img, y, image_path=image_path, img_type=img_type, transform=None)
    test_loader = DataLoader(test_set, batch_size=args.bh_size, shuffle=False)

    model = BrainAgeEstimator(args, device, channels, img_size, image_patch_size,
                              depth, depth_patch_size, x_dim)
    model = nn.DataParallel(model)
    # load pretrained weight
    if os.path.exists(trained_weight):
        print('load pretrained weights')
        model.load_state_dict(torch.load(trained_weight))
        print('load successfully.')

    model = model.module
    model.to(device)
    # compute saliency maps
    saliency_maps = compute_saliency_maps(model, test_loader)
    # calculate saliency probability maps
    saliency_probability_maps = compute_saliency_probability_maps(saliency_maps)
    print(np.sum(saliency_probability_maps))
    # average saliency probability maps
    saliency_probability_maps = np.mean(saliency_probability_maps, axis=0)
    # np.save(os.path.join(res_path, f'saliency_map.npy'), saliency_maps)
    np.save(os.path.join(res_path, f'mean_saliency_probability_map.npy'), saliency_probability_maps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_name', type=str, choices=['ADNI', 'UKBiobank'], default="UKBiobank")
    # model name
    parser.add_argument('--model_name', type=str, default='vit')
    parser.add_argument('--img_type', type=str, choices=['npy', 'nii'], default="npy")
    # healthy or not
    parser.add_argument('--is_healthy', type=int, default=1)
    # fold
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--dim_mlp', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=6)
    parser.add_argument('--dim_head', type=int, default=64)
    parser.add_argument('--attn_layers', type=int, default=3)
    # step 32
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--step', type=int, default=32)

    parser.add_argument('--hidden_dim_encoder', type=int, default=256)
    parser.add_argument('--encoder_layers', type=int, default=2)

    parser.add_argument('--pooling', type=str,
                        choices=['sum', 'mean'],
                        default='sum')

    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--emb_dropout', type=float, default=0.)

    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--k_fold', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--bh_size', type=int, default=1)

    parser.add_argument('--pretrain', type=bool, default=False)

    parser.add_argument('--seed', type=int, default=1234)

    main(parser.parse_args())
