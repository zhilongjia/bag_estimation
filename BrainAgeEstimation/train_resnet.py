import argparse
import os

import sys


import pandas as pd
import torch
from torch import optim

from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from utils.datasets import NewDataset, get_transform, TestDataset
from model.resnet3D import ResNet18
from utils.load_data import load_data_ukb
from utils.logger import Logger
from utils.utlis import seed_everything, mkdir_if_needed
from train_and_eval import train_and_evaluate, evaluate2

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(args.seed)

    cwd = os.getcwd()

    sys.stdout = Logger(filename=os.path.join(cwd, 'train_resnet.log'))

    data_path = os.path.join(cwd, 'data')
    image_path = os.path.join(data_path, 'npy_data')
    save_path = os.path.join(cwd, 'saved_models', 'resnet')
    res_path = os.path.join(cwd, 'results', 'resnet')

    mkdir_if_needed(save_path)
    mkdir_if_needed(res_path)

    # load data
    img, x, y = load_data_ukb(data_path, train=True)

    print(f'{len(img)} for training.')

    kf = KFold(n_splits=args.k_fold)
    tr_fold = 1
    for tr_idx, va_idx in kf.split(x):
        print('-' * 80 + f'training fold {tr_fold}' + '-' * 80)
        train_set = NewDataset(img[tr_idx], y[tr_idx], image_path=image_path, img_type='npy', transform=None)
        va_set = NewDataset(img[va_idx], y[va_idx], image_path=image_path, img_type='npy', transform=None)
        tr_loader = DataLoader(train_set, batch_size=args.bh_size, shuffle=False)
        va_loader = DataLoader(va_set, batch_size=args.bh_size, shuffle=False)
        print(f'{len(tr_loader.dataset)} for training, {len(va_loader.dataset)} for evaluation')

        if os.path.exists(os.path.join(save_path, f'model_vit_fold{tr_fold}.pth')):
            print('training complete...')
            model = ResNet18()
            model = model.to(device)
            if os.path.exists(os.path.join(res_path, f'pred_age_fold{tr_fold}.csv')):
                print(f'fold {tr_fold} already evaluated, continue...')
                tr_fold += 1
                continue
        else:
            print('training start...')
            model = ResNet18()
            # print model parameters size
            for name, param in model.named_parameters():
                print(name, param.size())

            model = model.to(device)

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
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--emb_dropout', type=float, default=0.)

    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--k_fold', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--bh_size', type=int, default=2)

    parser.add_argument('--seed', type=int, default=1234)

    main(parser.parse_args())
