import itertools
import os
import time

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from sklearn import metrics
from model.brainAgeEstimator2 import BrainAgeEstimator
from typing import Optional
from torch.utils.data import DataLoader
import logging
from sklearn.metrics import r2_score

from utils.utlis import to_device, save_model

EPS = 1e-8
INF = 1e8


def train_and_evaluate(args, device, model: BrainAgeEstimator, optimizer,
                       train_loader: DataLoader, test_loader: DataLoader,
                       tr_fold=0, save_path=None, save_name=None):
    # if isinstance(model, torch.nn.DataParallel):
    #     model = model.module

    tr_losses, tr_mae_all, tr_r2_all = [], [], []
    va_losses, va_mae_all, va_r2_all = [], [], []

    epoch_num = args.epochs

    # early stopping for preventing over-fitting
    tolerance = 0
    min_mae = INF

    # params for model
    state_dict = model.state_dict()

    for i in range(1, epoch_num + 1):
        # loss mse
        loss_total = 0.
        step = 0
        model.train()
        for data in train_loader:
            (bh_img, bh_y) = data
            bh_img, bh_y = to_device(device, bh_img, bh_y)

            optimizer.zero_grad()
            # out = model(bh_img, bh_x)
            if isinstance(model, torch.nn.DataParallel):
                out = model.module(bh_img)
            else:
                out = model(bh_img)
            loss = F.mse_loss(out, bh_y)
            loss.backward()
            optimizer.step()

            loss_total += loss.item()
            step += 1
        loss_total /= step

        # tr_mse, tr_mae = evaluate(args, model, device, train_loader)

        # used for plot
        tr_losses.append(loss_total)

        va_mse, va_mae, va_r2 = evaluate(args, model, device, test_loader)

        va_losses.append(va_mse)
        va_mae_all.append(va_mae)
        va_r2_all.append(va_r2)

        text = f'Train Epoch {i:03d} | loss_mse={loss_total:.4f}, var_mse={va_mse:.3f},' \
               f' var_mae={va_mae:.3f}, var_r2={va_r2:.3f}'

        print(text)
        print('-' * 100)

        if va_mae < min_mae:
            min_mae = va_mae
            # update params for model
            state_dict = model.state_dict()
            print('updating model...')
            # saving the params with the best var results
            if save_path is not None:
                save_model(state_dict, save_path, save_name=save_name)
            tolerance = 0
        else:
            tolerance += 1
            if tolerance == 10:
                break

    return state_dict


def evaluate(args, model: BrainAgeEstimator, device, loader: DataLoader, save: bool = False):

    # if isinstance(model, torch.nn.DataParallel):
    #     model = model.module

    model.eval()
    mse_total, mae_total = 0, 0
    step = 0
    preds = []
    trues = []
    with torch.no_grad():
        for data in loader:
            (bh_img, bh_y) = data
            bh_img, bh_y = to_device(device, bh_img, bh_y)
            # out = model(bh_img, bh_x)
            if isinstance(model, torch.nn.DataParallel):
                out = model.module(bh_img)
            else:
                out = model(bh_img)
            loss_mse = F.mse_loss(out, bh_y)
            loss_mae = F.l1_loss(out, bh_y)
            preds += out.detach().cpu().tolist()
            trues += bh_y.detach().cpu().tolist()
            mse_total += loss_mse.item()
            mae_total += loss_mae.item()
            step += 1

    mse = mse_total / step
    mae = mae_total / step
    r2 = r2_score(trues, preds)
    if not save:
        return mse, mae, r2
    else:
        return mse, mae, r2, preds


def evaluate2(args, model: BrainAgeEstimator, device, loader: DataLoader):

    # if isinstance(model, torch.nn.DataParallel):
    #     model = model.module

    model.eval()
    mse_total, mae_total = 0, 0
    step = 0
    preds = []
    trues = []
    ids = []
    with torch.no_grad():
        for data in loader:
            (bh_img, bh_y, bh_id) = data
            bh_img, bh_y = to_device(device, bh_img, bh_y)
            # out = model(bh_img, bh_x)
            if isinstance(model, torch.nn.DataParallel):
                out = model.module(bh_img)
            else:
                out = model(bh_img)

            loss_mse = F.mse_loss(out, bh_y)
            loss_mae = F.l1_loss(out, bh_y)
            preds += out.view(-1).detach().cpu().tolist()
            trues += bh_y.view(-1).detach().cpu().tolist()
            if isinstance(bh_id, Tensor):
                ids += bh_id.detach().cpu().tolist()
            else:
                ids += list(bh_id)
            mse_total += loss_mse.item()
            mae_total += loss_mae.item()
            step += 1

    mse = mse_total / step
    mae = mae_total / step
    r2 = r2_score(trues, preds)

    return mse, mae, r2, preds, ids


