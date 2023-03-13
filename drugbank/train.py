import gc
import torch
gc.collect()
torch.cuda.empty_cache()

import os
import time
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim

from utils import *
from metrics import *
from model import LSFC
from config import BIN_config_DBPE
from dataset import load_ddi_dataset
from log.train_logger import TrainLogger

import warnings
warnings.filterwarnings("ignore")

test1 = 0

def test(model, criterion, test_loader, device):
    model.eval()
    running_loss = AverageMeter()

    pred_list = []
    label_list = []
    for data in test_loader:
        test1 = 1
        head_pairs, tail_pairs, rel, label, d_v, p_v, input_mask_d, input_mask_p = [d.to(device) for d in data]

        with torch.no_grad():
            pred = model((head_pairs, tail_pairs, rel, d_v, p_v, input_mask_d, input_mask_p), test1)
            loss = criterion(pred, label)
            pred_cls = torch.sigmoid(pred)
            pred_list.append(pred_cls.view(-1).detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())
            running_loss.update(loss.item(), label.size(0))

    pred_probs = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    acc, auroc, f1_score, precision, recall, ap = do_compute_metrics(pred_probs, label)

    epoch_loss = running_loss.get_average()
    running_loss.reset()
    model.train()
    return epoch_loss, acc, auroc, f1_score, precision, recall, ap


def val(model, criterion, dataloader, device):
    model.eval()
    running_loss = AverageMeter()

    pred_list = []
    label_list = []
    for data in dataloader:
        test1 = 0
        head_pairs, tail_pairs, rel, label, d_v, p_v, input_mask_d, input_mask_p = [d.to(device) for d in data]
        with torch.no_grad():
            pred = model((head_pairs, tail_pairs, rel, d_v, p_v, input_mask_d, input_mask_p), test1)
            loss = criterion(pred, label)
            pred_cls = torch.sigmoid(pred)
            pred_list.append(pred_cls.view(-1).detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())
            running_loss.update(loss.item(), label.size(0))

    pred_probs = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    acc, auroc, f1_score, precision, recall, ap = do_compute_metrics(pred_probs, label)

    epoch_loss = running_loss.get_average()
    running_loss.reset()

    model.train()
    return epoch_loss, acc, auroc, f1_score, precision, recall, ap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=0, help='[0, 1, 2]')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--hidden_size', type=int, default=384, help='number of hidden size')
    parser.add_argument('--save_model', action='store_true', help='whether save model or not')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='number of epochs')
    args = parser.parse_args()

    params = dict(
        model='LSFC',
        data_root='data/preprocessed/',
        save_dir='save',
        dataset='drugbank',
        fold=args.fold,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        save_model=args.save_model,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    logger = TrainLogger(params)
    logger.info(__file__)

    fold = params.get('fold')
    epochs = params.get('epochs')
    batch_size = params.get('batch_size')
    hidden_size = params.get('hidden_size')
    save_model = params.get('save_model')
    lr = params.get('lr')
    weight_decay = params.get('weight_decay')
    data_root = params.get('data_root')
    data_set = params.get('dataset')
    data_path = os.path.join(data_root, data_set)

    train_loader, val_loader, test_loader = load_ddi_dataset(root=data_path, batch_size=batch_size, fold=fold)
    device = torch.device('cuda:0')

    config = BIN_config_DBPE()
    model = LSFC(hidden_size, **config).cuda()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** (epoch))

    running_loss = AverageMeter()
    running_acc = AverageMeter()

    model.train()

    for epoch in range(epochs):
        begin_time = time.time()
        for data in train_loader:
            head_pairs, tail_pairs, rel, label, d_v, p_v, input_mask_d, input_mask_p = [d.to(device) for d in data]
            pred = model((head_pairs, tail_pairs, rel, d_v, p_v, input_mask_d, input_mask_p))
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred_cls = (torch.sigmoid(pred) > 0.5).detach().cpu().numpy()
            acc = accuracy(label.detach().cpu().numpy(), pred_cls)
            running_acc.update(acc)
            running_loss.update(loss.item(), label.size(0))

        end_time = time.time()
        run_time = round(end_time - begin_time)
        hour = run_time // 3600
        minute = (run_time - 3600 * hour) // 60
        second = run_time - 3600 * hour - 60 * minute
        print(f'当前epoch运行时间：{hour}小时{minute}分钟{second}秒')

        epoch_loss = running_loss.get_average()
        epoch_acc = running_acc.get_average()
        running_loss.reset()
        running_acc.reset()

        val_loss, val_acc, val_auroc, val_f1_score, val_precision, val_recall, val_ap = val(model, criterion, val_loader, device)
        msg = "epoch-%d, train_loss-%.4f, train_acc-%.4f, val_loss-%.4f, val_acc-%.4f, val_auroc-%.4f, val_f1_score-%.4f, val_prec-%.4f, val_rec-%.4f, val_ap-%.4f" % (epoch, epoch_loss, epoch_acc, val_loss, val_acc, val_auroc, val_f1_score, val_precision, val_recall, val_ap)
        logger.info(msg)

        scheduler.step()

        if save_model:
            msg = "epoch-%d, train_loss-%.4f, train_acc-%.4f, val_loss-%.4f, val_acc-%.4f" % (epoch, epoch_loss, epoch_acc, val_loss, val_acc)
            # del_file(logger.get_model_dir())
            save_model_dict(model, logger.get_model_dir(), msg)

    test_loss, test_acc, test_auroc, test_f1_score, test_precision, test_recall, test_ap = test(model, criterion, test_loader, device)
    msg = "test_loss-%.4f, test_acc-%.4f, test_auroc-%.4f, test_f1_score-%.4f, test_prec-%.4f, test_rec-%.4f, test_ap-%.4f" % (test_loss, test_acc, test_auroc, test_f1_score, test_precision, test_recall, test_ap)
    logger.info(msg)


if __name__ == "__main__":
    main()