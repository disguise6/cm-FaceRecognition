#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/5/12 14:44
# @Author  : PPq
from __future__ import print_function
import argparse
import os
import shutil
import time
from torch.nn import DataParallel
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.nn.parameter import Parameter
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from core import IDR
from core.utils import init_log
from dataloader.CASIA_NIR_VIS_IDR2 import CASIA_NIR_VIS
import numpy as np
from load_imglist import ImageList
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch Light CNN Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='LightCNN')
parser.add_argument('--cuda', '-c', default=True)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--epochs', default=70, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default:s 128)')
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--model', default='IDRnet', type=str, metavar='Model',
                    help='model type: IDRnet')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrain', default='model/LightCNN9.tar', type=str, metavar='PATH',
                    help='path to pretrained model for basic network(default:none)')
parser.add_argument('--root_path', default='', type=str, metavar='PATH',
                    help='path to root path of images (default: none)')
parser.add_argument('--train_dir',
                    default='/mnt/traffic/fkq/NIR_VIS_Face_Recognition-master/dataset/CASIA-NIR-VIS-2.0/NIR-VIS-2.0', type=str,
                    metavar='PATH',
                    help='path to training list (default: none)')
parser.add_argument('--val_dir', default='/mnt/traffic/fkq/NIR_VIS_Face_Recognition-master/dataset/CASIA-NIR-VIS-2.0/NIR-VIS-2.0',
                    type=str, metavar='PATH',
                    help='path to validation list (default: none)')
parser.add_argument('--save_path', default='model/tmp/', type=str, metavar='PATH',
                    help='path to save checkpoint (default: none)')
parser.add_argument('--num_classes', default=321, type=int,
                    metavar='N', help='number of classes')


def main():
    global args
    args = parser.parse_args()

    # create Light CNN for face recognition
    if args.model == 'IDRnet':
        model = IDR.IDRnet(num_classes=args.num_classes)
        print('use IDRnet')
    else:
        print('Error model type\n')

    
    print(torch.cuda.current_device())
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    device_ids = [0, 1, 2, 3]
    if args.cuda:
        model = torch.nn.DataParallel(model, device_ids).cuda()
  
    '''
    print(model.module)
    for name, value in model.named_parameters():
        print(name, value.shape)
    '''
    
    # large lr for last fc parameters
    basic_params = []
    for name, value in model.module.basic_layer.named_parameters():
        if 'bias' in name:
            basic_params += [{'params': value, 'lr': 0.001 * args.lr, 'weight_decay': 0}]  #2
        else:
            basic_params += [{'params': value, 'lr': 0.001 * args.lr}] #1
    
    basic_opt = torch.optim.SGD(basic_params, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # large lr for last fc parameters
    feat_params = []
    for name, value in model.module.feature_layer.named_parameters():
        if 'bias' in name:
            if 'fc' in name: #fc2.bias
                print('fc + bias : ' + name)
                feat_params += [{'params': value, 'lr': 0.95 * args.lr, 'weight_decay': 0}]
            else: #bias
                print('bias : ' + name)
                feat_params += [{'params': value, 'lr': 0.95 * args.lr, 'weight_decay': 0}]
        else:  
            if 'fc' in name: #fc.weight
                print('fc + weight : ' + name)
                feat_params += [{'params': value, 'lr': 0.95 * args.lr}]
            else:  #weight
                print('weight:' + name)
                feat_params += [{'params': value, 'lr': 0.95 * args.lr}]
    feat_opt = torch.optim.SGD(feat_params, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    optimizer = [basic_opt, feat_opt]
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            #args.start_epoch = checkpoint['epoch']
            args.start_epoch = 0
            checkpoint = torch.load(args.resume)
            if 'LightCNN9' in args.resume:
                del(checkpoint['state_dict']['module.fc1.filter.weight'])
                del(checkpoint['state_dict']['module.fc1.filter.bias'])
                del(checkpoint['state_dict']['module.fc2.weight'])
                del(checkpoint['state_dict']['module.fc2.bias'])
                name = [key[7:] for key in checkpoint['state_dict'].keys()]
                value = checkpoint['state_dict'].values()
                new_state_dict = dict(zip(name, value))
                #print(list(checkpoint['state_dict'].keys()))
                model_dict = model.module.basic_layer.state_dict()
                model_dict.update(new_state_dict)
                model.module.basic_layer.load_state_dict(model_dict)
            #print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.pretrain:
        if os.path.isfile(args.pretrain):
            print("=> loading pretrain model for basic model '{}'".format(args.pretrain))
            checkpoint = torch.load(args.pretrain)
            if 'LightCNN9' in args.pretrain:
                del(checkpoint['state_dict']['module.fc1.filter.weight'])
                del(checkpoint['state_dict']['module.fc1.filter.bias'])
                del(checkpoint['state_dict']['module.fc2.weight'])
                del(checkpoint['state_dict']['module.fc2.bias'])
                name = [key[7:] for key in checkpoint['state_dict'].keys()]
                value = checkpoint['state_dict'].values()
                new_state_dict = dict(zip(name, value))
                #print(list(checkpoint['state_dict'].keys()))
                model_dict = model.module.basic_layer.state_dict()
                model_dict.update(new_state_dict)
                model.module.basic_layer.load_state_dict(model_dict)
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrain))

    weight_init(model)
    cudnn.benchmark = True

    # load image
    all_transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    # define trainloader and testloader
    trainset = CASIA_NIR_VIS(root=args.train_dir, transform=all_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=8, drop_last=False)

    valset = CASIA_NIR_VIS(root=args.val_dir, transform=all_transform, test=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=32,
                                             shuffle=False, num_workers=8, drop_last=False)

    # define loss function and optimizer
    basic_criterion = nn.CrossEntropyLoss()
    total_critertion = IDRLoss()
    criterion = [basic_criterion, total_critertion]
    if args.cuda:
        for cri in criterion:
            cri.cuda()

    validate(val_loader, model, criterion)
    loss_list = []
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer[0], epoch)
        adjust_learning_rate(optimizer[1], epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, loss_list)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        save_name = args.save_path + 'lightCNN_' + str(epoch + 1) + '_checkpoint.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'prec1': prec1,
        }, save_name)
    plt.plot(np.arange(args.start_epoch, args.epochs).astype(np.str), loss_list,  marker = 'o', color = 'r')
    plt.xlabel('epochs')
    plt.ylabel('train loss')
    plt.savefig('pci.png')


def train(train_loader, model, criterion, optimizer, epoch, loss_list):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target, idt) in enumerate(train_loader):
        data_time.update(time.time() - end)
        input = input.cuda()
        # print(input.shape)
        target = target.cuda()
        idt = idt.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        idt_var = torch.autograd.Variable(idt)
        # compute output
        output, _, _ = model(input_var, idt_var)
        basic_loss = criterion[0](output, target_var)

        # compute gradient and do SGD step
        optimizer[0].zero_grad()
        basic_loss.backward()
        optimizer[0].step()

        pn = model.module.feature_layer.unique_mfm1.filter.weight
        pv = model.module.feature_layer.unique_mfm2.filter.weight
        w = model.module.feature_layer.shared_mfm.filter.weight
        output, _, _ = model(input_var, idt_var)
        loss = criterion[1](output, target_var, w, pn, pv)
        optimizer[1].zero_grad()
        loss.backward()
        optimizer[1].step()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        # print('\nprec1:{}, prec5:{}'.format(prec1, prec5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
    loss_list.append(losses.avg)

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, idt) in enumerate(val_loader):
        input = input.cuda()
        #print('input.shape : {}'.format(input.shape))
        target = target.cuda()
        idt = idt.cuda()
        input_var = input.clone().detach()
        target_var = target.clone().detach()
        idt_var = idt.clone().detach()

        # compute output
        output, _, _ = model(input_var, idt_var)
        #print('output.shape: {}'.format(output.shape))
        pn = model.module.feature_layer.unique_mfm1.filter.weight
        #print('pn.shape: {}'.format(pn.shape))
        pv = model.module.feature_layer.unique_mfm2.filter.weight
        #print('pv.shape: {}'.format(pv.shape))
        w = model.module.feature_layer.shared_mfm.filter.weight
        loss = criterion[1](output, target_var, w, pn, pv)
        #loss = criterion[0](output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        # print('\nprec1:{}, prec5:{}'.format(prec1, prec5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

    print('\ntop1.sum: {}, top1.count: {}'.format(top1.sum, top1.count))
    print('\nTest set: Average loss: {}, Accuracy: ({})\n'.format(losses.avg, top1.avg))

    return top1.avg

def save_checkpoint(state, filename):
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    #print(target.shape)
    # (batch, feature)，从每个feature选择概率topk的类别
    _, pred = output.topk(maxk, 1, True, True)
    # 转置
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    #print(correct)
    res = []
    for k in topk:
        # 计算有多少个预测正确
        correct_k = correct[:k].contiguous()
        correct_k = correct_k.view(-1)
        #print(correct_k)
        correct_k = correct_k.float()
        correct_k = correct_k.sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def weight_init(model):
    #print(type(model.named_parameters()))
    dim = 256
    for name, param in model.named_parameters():
        if 'mfm' in name and 'weight' in name:
            print(name)
            nn.init.uniform_(param, -1.0/(dim ** 0.5), 1.0 / (dim ** 0.5))

def adjust_learning_rate(optimizer, epoch):
    scale = 0.457305051927326
    step  = 35
    lr = args.lr * (scale ** (epoch // step))
    print('lr: {}'.format(lr))
    if (epoch != 0) & (epoch % step == 0):
        print('Change lr')
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * scale


class IDRLoss(nn.Module):
    def __init__(self, w, pn, pv, lamda= 2):
        super(IDRLoss, self).__init__()
        self.lamda = lamda
        self.w = Parameter(torch.Tensor(w))
        self.pn = Parameter(torch.Tensor(pn))
        self.pv = Parameter(torch.Tensor(pv))
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, x, target):
        loss1 = self.ce_loss(x, target)
        loss2 = self.lamda * (torch.norm(torch.mm(self.pn.t(), self.w), p='fro') ** 2) + \
                        self.lamda * (torch.norm(torch.mm(self.pv.t(), self.w), p='fro') ** 2)
        print('loss1 : {}, loss2: {}'.format(loss1, loss2))
        return loss1 + loss2
        
if __name__ == '__main__':
    main()