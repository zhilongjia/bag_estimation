import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from utils.datasets import NewDataset, get_transform, TestDataset
from model.brainAgeEstimator2 import BrainAgeEstimator
from utils.load_data import load_data_ukb
from utils.logger import Logger
from utils.utlis import seed_everything, mkdir_if_needed
from train_and_eval import train_and_evaluate, evaluate2

os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(args.seed)

    cwd = os.getcwd()

    sys.stdout = Logger(filename=os.path.join(cwd, 'train_vit.log'))

    data_path = os.path.join(cwd, 'data')
    image_path = os.path.join(data_path, 't1_data')
    save_path = os.path.join(cwd, 'saved_models')
    res_path = os.path.join(cwd, 'results')

    mkdir_if_needed(res_path)

    # load data
    img, x, y = load_data_ukb(data_path, train=True)

    print(f'{len(img)} for training.')

    x_dim = x.shape[1]
    channels, depth, img_size = 1, 128, (128, 128)
    image_patch_size, depth_patch_size = 16, 16

    kf = KFold(n_splits=5)
    tr_fold = 1
    for tr_idx, va_idx in kf.split(x):
        print('-' * 80 + f'training fold {tr_fold}' + '-' * 80)
        train_set = NewDataset(img[tr_idx], y[tr_idx], image_path=image_path, img_type='nii', transform=get_transform())
        va_set = NewDataset(img[va_idx], y[va_idx], image_path=image_path, img_type='nii', transform=get_transform())
        tr_loader = DataLoader(train_set, batch_size=args.bh_size, shuffle=False)
        va_loader = DataLoader(va_set, batch_size=args.bh_size, shuffle=False)
        print(f'{len(tr_loader.dataset)} for training, {len(va_loader.dataset)} for evaluation')

        model = BrainAgeEstimator(args, device, channels, img_size, image_patch_size,
                                  depth, depth_patch_size, x_dim)

        if torch.cuda.device_count() > 1:
            print(torch.cuda.device_count(), "GPUs are used.")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            model = nn.DataParallel(model)
        model = model.to(device)

        # load pretrained weight
        # if args.pretrain and os.path.exists(pretrained_weight):
        #     print('load pretrained weights')
        #     model.module.load_state_dict(torch.load(pretrained_weight))
        #     print('load successfully.')

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        train_and_evaluate(args, device, model, optimizer, tr_loader, va_loader, tr_fold=tr_fold,
                           save_path=save_path, save_name=f'model_vit_fold{tr_fold}')

        model.load_state_dict(torch.load(os.path.join(save_path, f'model_vit_fold{tr_fold}.pth')))
        # model.load_state_dict(state_dict)
        va_set2 = TestDataset(img[va_idx], y[va_idx], image_path=image_path, img_type='npy')
        va_loader2 = DataLoader(va_set2, batch_size=args.bh_size, shuffle=False)
        mse, mae, r2, preds, ids = evaluate2(args, model, device, va_loader2)
        text = f'Training fold {tr_fold} final results: | va_mse={mse:.3f}, va_mae={mae:.3f}, va_r2={r2:.3f}'
        print(text)

        # save pred result
        df = pd.DataFrame({
            'Eid': ids,
            'pred_age': preds
        })
        df.to_csv(os.path.join(res_path, f'pred_age_fold{tr_fold}.csv'), index=False)

        tr_fold += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_name', type=str, choices=['ADNI', 'UKBiobank'], default="UKBiobank")
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--dim_mlp', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=6)
    parser.add_argument('--dim_head', type=int, default=64)
    parser.add_argument('--attn_layers', type=int, default=3)

    parser.add_argument('--hidden_dim_encoder', type=int, default=256)
    parser.add_argument('--encoder_layers', type=int, default=2)

    parser.add_argument('--pooling', type=str,
                        choices=['sum', 'mean'],
                        default='sum')

    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--emb_dropout', type=float, default=0.)

    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--k_fold', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--bh_size', type=int, default=32)

    parser.add_argument('--pretrain', type=bool, default=False)

    parser.add_argument('--seed', type=int, default=1234)

    main(parser.parse_args())
