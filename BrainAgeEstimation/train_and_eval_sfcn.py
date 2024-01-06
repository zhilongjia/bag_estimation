from tqdm import tqdm
import numpy as np
import torch
from torch import Tensor, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from model.sfcn.loss import my_KLDivLoss, num2vect

from utils.utlis import to_device, save_model

EPS = 1e-8
INF = 1e8


def train_and_evaluate(args, device, model, optimizer,
                       train_loader: DataLoader, test_loader: DataLoader,
                       tr_fold=0, save_path=None, save_name=None):
    # if isinstance(model, torch.nn.DataParallel):
    #     model = model.module

    train_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    tr_losses, tr_mae_all, tr_r2_all = [], [], []
    va_losses, va_mae_all, va_r2_all = [], [], []

    epoch_num = args.epochs

    # early stopping for preventing over-fitting
    tolerance = 0
    min_mae = INF

    # params for model
    state_dict = model.state_dict()

    # use progress bar tqdm
    for i in range(1, epoch_num + 1):
        # loss mse
        loss_total = 0.
        step = 0
        model.train()
        loss_func = my_KLDivLoss()
        for data in tqdm(train_loader, desc=f'Training Epoch {i:03d}'):
            (bh_img, bh_y) = data
            bh_img, bh_y = to_device(device, bh_img, bh_y)
            bin_range = [40, 85]
            bin_step = 1
            sigma = 1

            y, bc = num2vect(bh_y.detach().cpu().numpy(), bin_range, bin_step, sigma)
            y = torch.as_tensor(y, dtype=torch.float32).to(device)

            optimizer.zero_grad()
            # out = model(bh_img, bh_x)
            if isinstance(model, torch.nn.DataParallel):
                out = model.module(bh_img)
            else:
                out = model(bh_img)

            # loss = F.mse_loss(out, bh_y)
            N_bh = out[0].size(0)
            x = out[0].reshape(N_bh, -1)

            loss = loss_func(x, y)

            loss.backward()
            optimizer.step()

            loss_total += loss.item()
            step += 1

        loss_total /= step

        train_scheduler.step()

        # used for plot
        tr_losses.append(loss_total)

        val_kld_loss, va_mse, va_mae, va_r2 = evaluate(args, model, device, test_loader)

        va_losses.append(va_mse)
        va_mae_all.append(va_mae)
        va_r2_all.append(va_r2)

        text = (f'Train Epoch {i:03d} | loss_KLD={loss_total:.3f},'
                f' var_KLD={val_kld_loss:.3f}, var_mse={va_mse:.3f},'
                f' var_mae={va_mae:.3f}, var_r2={va_r2:.3f}')

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


def evaluate(args, model, device, loader: DataLoader, save: bool = False):

    # if isinstance(model, torch.nn.DataParallel):
    #     model = model.module

    model.eval()
    var_loss_total, mse_total, mae_total = 0, 0, 0
    step = 0
    preds = []
    trues = []
    loss_func = my_KLDivLoss()
    with torch.no_grad():
        for data in tqdm(loader, desc=f'Evaluating'):
            (bh_img, bh_y) = data
            bh_img, bh_y = to_device(device, bh_img, bh_y)
            # out = model(bh_img, bh_x)
            if isinstance(model, torch.nn.DataParallel):
                out = model.module(bh_img)
            else:
                out = model(bh_img)

            bin_range = [40, 85]
            bin_step = 1
            sigma = 1
            y, bc = num2vect(bh_y.detach().cpu().numpy(), bin_range, bin_step, sigma)
            y = torch.as_tensor(y, dtype=torch.float32).cuda(non_blocking=True)
            N_bh = out[0].size(0)

            x = out[0].reshape([N_bh, -1])
            val_loss = loss_func(x, y)

            x = x.cpu().numpy().reshape(N_bh, -1)
            prob = np.exp(x)
            pred = prob @ bc
            true_ages = bh_y.detach().cpu().numpy()

            loss_mse = mean_squared_error(pred, true_ages)
            loss_mae = mean_absolute_error(pred, true_ages)

            # preds to list
            preds += pred.tolist()
            trues += true_ages.tolist()

            mse_total += loss_mse
            mae_total += loss_mae
            var_loss_total += val_loss.item()
            step += 1

    mse = mse_total / step
    mae = mae_total / step
    kld = var_loss_total / step

    r2 = r2_score(trues, preds)
    if not save:
        return kld, mse, mae, r2
    else:
        return kld, mse, mae, r2, preds


def evaluate2(args, model, device, loader: DataLoader):

    # if isinstance(model, torch.nn.DataParallel):
    #     model = model.module

    model.eval()
    mse_total, mae_total = 0, 0
    step = 0
    preds = []
    trues = []
    ids = []
    loss_func = my_KLDivLoss()

    with torch.no_grad():
        for data in tqdm(loader, desc=f'Evaluating'):
            (bh_img, bh_y, bh_id) = data
            bh_img, bh_y = to_device(device, bh_img, bh_y)
            # out = model(bh_img, bh_x)
            if isinstance(model, torch.nn.DataParallel):
                out = model.module(bh_img)
            else:
                out = model(bh_img)

            bin_range = [40, 85]
            bin_step = 1
            sigma = 1
            y, bc = num2vect(bh_y.detach().cpu().numpy(), bin_range, bin_step, sigma)
            y = torch.as_tensor(y, dtype=torch.float32).cuda(non_blocking=True)

            N_bh = out[0].size(0)

            x = out[0].reshape([N_bh, -1])
            val_loss = loss_func(x, y)

            x = x.cpu().numpy().reshape(N_bh, -1)
            prob = np.exp(x)
            pred = prob @ bc
            true_ages = bh_y.detach().cpu().numpy()

            loss_mse = mean_squared_error(pred, true_ages)
            loss_mae = mean_absolute_error(pred, true_ages)
            preds += pred.tolist()
            trues += true_ages.tolist()

            if isinstance(bh_id, Tensor):
                ids += bh_id.detach().cpu().tolist()
            else:
                ids += list(bh_id)

            mse_total += loss_mse
            mae_total += loss_mae
            step += 1

    mse = mse_total / step
    mae = mae_total / step
    r2 = r2_score(trues, preds)

    return mse, mae, r2, preds, ids


