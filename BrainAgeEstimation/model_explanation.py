import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.datasets import TestDataset, MaskedDataset
from model.brainAgeEstimator import BrainAgeEstimator
from utils.load_data import load_data_ukb
from utils.logger import Logger
from utils.utlis import seed_everything,normalize_data
from train_and_eval import train_and_evaluate, evaluate2

os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(args.seed)

    cwd = os.getcwd()

    data_path = os.path.join(cwd, 'data')
    image_path = os.path.join(data_path, 't1_data')

    # load test data
    # BAG data
    df = pd.read_csv('results/mae_bagm3.csv')
    img = np.asarray(df['Eid'])
    y = np.asarray(df['age'])
    x = np.asarray(df.iloc[:, 12:36])
    x = normalize_data(x, norm_mode='standard')

    print(f'{len(img)} for testing.')

    x_dim = x.shape[1]
    channels, depth, img_size = 1, 128, (128, 128)
    image_patch_size, depth_patch_size = 16, 16

    trained_weight = os.path.join(cwd, 'saved_models', 'model_vit_fold1.pth')

    test_set = MaskedDataset(img, y, image_path=image_path, img_type='nii', mask=None)
    test_loader = DataLoader(test_set, batch_size=args.bh_size, shuffle=False)

    model = BrainAgeEstimator(args, device, channels, img_size, image_patch_size,
                            depth, depth_patch_size, x_dim)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    # load pretrained weight
    if os.path.exists(trained_weight):
        print('load pretrained weights')
        xxx = torch.load(trained_weight)
        model.load_state_dict(xxx)
        print('load successfully.')

    mse, mae_normal, r2, preds, ids = evaluate2(args, model, device, test_loader)

    delta_mae = np.zeros((12,12,12))
    iter = [0,0,0]
    for it1 in range(12):
        for it2 in range(12):
            for it3 in range(12):
                iter = [it1,it2,it3]
                print('iter:')
                print(iter)
                mask = np.ones((182,218,182))
                mask[1+iter[0]*15:1+(iter[0]+1)*15,1+iter[1]*18:1+(iter[1]+1)*18,1+iter[2]*15:1+(iter[2]+1)*15] = np.zeros((15,18,15))
                test_set = MaskedDataset(img, y, image_path=image_path, img_type='nii', mask=mask)
                test_loader = DataLoader(test_set, batch_size=args.bh_size, shuffle=False)

                model = BrainAgeEstimator(args, device, channels, img_size, image_patch_size,
                                        depth, depth_patch_size, x_dim)

                if torch.cuda.device_count() > 1:
                    print("Let's use", torch.cuda.device_count(), "GPUs!")
                    model = nn.DataParallel(model)
                model = model.to(device)

                # load pretrained weight
                if os.path.exists(trained_weight):
                    print('load pretrained weights')
                    xxx = torch.load(trained_weight)
                    model.load_state_dict(xxx)
                    print('load successfully.')

                mse, mae, r2, preds, ids = evaluate2(args, model, device, test_loader)
                print(mae - mae_normal)
                # save result
                delta_mae[it1][it2][it3] = mae - mae_normal
    print(delta_mae)
    np.save(os.path.join('shap_values', 'bagm3_mae.npy'), delta_mae)


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

    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--emb_dropout', type=float, default=0.)

    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--k_fold', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--bh_size', type=int, default=1)

    parser.add_argument('--seed', type=int, default=1234)

    main(parser.parse_args())
