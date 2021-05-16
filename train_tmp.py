#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/27 11:19
# @Author  : PPq
import os
import torch.utils.data
from torch import nn
from torch.nn import DataParallel
from datetime import datetime
from config import BATCH_SIZE, SAVE_FREQ, RESUME, SAVE_DIR, TEST_FREQ, TOTAL_EPOCH, MODEL_PRE, GPU
from config import lr, momentum, weight_decay
from config import CASIA_DATA_DIR, LFW_DATA_DIR
from core import light_cnn
from core.utils import init_log
from dataloader.CASIA_NIR_VIS import CASIA_NIR_VIS
from dataloader.LFW_loader import LFW
from torch.optim import lr_scheduler
import torch.optim as optim
import time
from lfw_eval import parseList, evaluation_10_fold
import numpy as np
import scipy.io
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def adjust_learning_rate(optimizer, epoch):
    scale = 0.457305051927326
    step = 10
    lr1 = lr * (scale ** (epoch // step))
    print('lr1: {}'.format(lr1))
    if (epoch != 0) & (epoch % step == 0):
        print('Change lr')
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * scale


# gpu init
gpu_list = ''
multi_gpus = False
if isinstance(GPU, int):
    gpu_list = str(GPU)
else:
    multi_gpus = True
    for i, gpu_id in enumerate(GPU):
        gpu_list += str(gpu_id)
        if i != len(GPU) - 1:
            gpu_list += ','
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list

# other init
start_epoch = 1
save_dir = os.path.join(SAVE_DIR, MODEL_PRE + 'v2_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
if os.path.exists(save_dir):
    raise NameError('model dir exists!')
os.makedirs(save_dir)
logging = init_log(save_dir)
_print = logging.info

all_transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.ToTensor(),
])
# define trainloader and testloader
trainset = CASIA_NIR_VIS(root=CASIA_DATA_DIR, transform=all_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=8, drop_last=False)

# nl: left_image_path
# nr: right_image_path
nl, nr, folds, flags = parseList(root=LFW_DATA_DIR, transform=all_transform)
testdataset = CASIA_NIR_VIS(root=CASIA_DATA_DIR, transform=all_transform)
testloader = torch.utils.data.DataLoader(testdataset, batch_size=32,
                                         shuffle=False, num_workers=8, drop_last=False)

# define model
net = light_cnn.LightCNN_29Layers()

if RESUME:
    ckpt = torch.load(RESUME)
    net.load_state_dict(ckpt['net_state_dict'])
    start_epoch = ckpt['epoch'] + 1

# define optimizers
# large lr for last fc parameters
params = []
for name, value in net.named_parameters():
    if 'bias' in name:
        if 'fc2' in name:
            params += [{'params': value, 'lr': 20 * lr, 'weight_decay': 0}]
        else:
            params += [{'params': value, 'lr': 2 * lr, 'weight_decay': 0}]
    else:
        if 'fc2' in name:
            params += [{'params': value, 'lr': 10 * lr}]
        else:
            params += [{'params': value, 'lr': 1 * lr}]

optimizer = torch.optim.SGD(params, lr,
                            momentum=momentum,
                            weight_decay=weight_decay)

net = net.cuda()
# ArcMargin = ArcMargin.cuda()
if multi_gpus:
    net = DataParallel(net)
    # ArcMargin = DataParallel(ArcMargin)
criterion = torch.nn.CrossEntropyLoss()

best_acc = 0.0
best_epoch = 0
for epoch in range(start_epoch, TOTAL_EPOCH + 1):
    adjust_learning_rate(optimizer, epoch)
    # train model
    _print('Train Epoch: {}/{} ...'.format(epoch, TOTAL_EPOCH))
    net.train()

    train_total_loss = 0.0
    total = 0
    since = time.time()
    for data in trainloader:
        img, label = data[0].cuda(), data[1].cuda()
        # print(img.shape)
        batch_size = img.size(0)
        optimizer.zero_grad()

        raw_logits, _ = net(img)

        # output = ArcMargin(raw_logits, label)
        total_loss = criterion(raw_logits, label)
        total_loss.backward()
        optimizer.step()

        train_total_loss += total_loss.item() * batch_size
        total += batch_size

    train_total_loss = train_total_loss / total
    time_elapsed = time.time() - since
    loss_msg = '    total_loss: {:.4f} time: {:.0f}m {:.0f}s' \
        .format(train_total_loss, time_elapsed // 60, time_elapsed % 60)
    _print(loss_msg)

    # test model on lfw
    if epoch % TEST_FREQ == 0:
        net.eval()
        featureLs = None
        featureRs = None
        _print('Test Epoch: {} ...'.format(epoch))
        for data in testloader:
            for i in range(len(data)):
                data[i] = data[i].cuda()
            res = [net(d)[0].data.cpu().numpy() for d in data]
            featureL = np.concatenate((res[0], res[1]), 1)
            featureR = np.concatenate((res[2], res[3]), 1)
            if featureLs is None:
                featureLs = featureL
            else:
                featureLs = np.concatenate((featureLs, featureL), 0)
            if featureRs is None:
                featureRs = featureR
            else:
                featureRs = np.concatenate((featureRs, featureR), 0)

        result = {'fl': featureLs, 'fr': featureRs, 'fold': folds, 'flag': flags}
        # save tmp_result
        scipy.io.savemat('./result/tmp_result.mat', result)
        accs = evaluation_10_fold('./result/tmp_result.mat')
        _print('    ave: {:.4f}'.format(np.mean(accs) * 100))

    # save model
    if epoch % SAVE_FREQ == 0:
        msg = 'Saving checkpoint: {}'.format(epoch)
        _print(msg)
        if multi_gpus:
            net_state_dict = net.module.state_dict()
        else:
            net_state_dict = net.state_dict()
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        torch.save({
            'epoch': epoch,
            'net_state_dict': net_state_dict},
            os.path.join(save_dir, '%03d.ckpt' % epoch))
print('finishing training')
