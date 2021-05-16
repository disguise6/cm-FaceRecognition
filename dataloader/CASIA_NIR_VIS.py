#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/5/3 13:00
# @Author  : PPq
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import os
import torch
import glob2


def read_list(list_path):
    img_list = []
    with open(list_path, 'r') as f:
        for line in f.readlines()[0:]:
            img_path = line.strip().split()
            img_list.append(img_path[0])
    print('There are {} images..'.format(len(img_list)))
    return img_list


def revise_name(probe_img_name):
    # print(probe_img_name, type(probe_img_name))
    suffix = probe_img_name.split('.')
    if suffix[-1] != 'bmp':
        suffix[-1] = 'bmp'

    probe_img_name = '.'.join(suffix)
    revise_name = probe_img_name.split('\\')

    # print(revise_name)
    #revise_name[1] += '_128x128'  # sgong
    temp = ""
    for i in range(len(revise_name)):
        temp = temp + revise_name[i]
        if i != len(revise_name) - 1:
            temp += '\\'
    return temp

class CASIA_NIR_VIS(Dataset):
    def __init__(self, root, transform=None, test=False):
        self.root = root
        self.transform = transform
        self.file_txt = ''
        self.file_list = []

        protocols = 'MyProtocols'
        if test == False:
            file_txt = r'nir_train.txt'
            file_list = glob2.glob(root + '/' + protocols + '/'+ file_txt)
        else:
            file_txt = r'nir_train_dev.txt'
            file_list = glob2.glob(root + '/' + protocols + '/'+ file_txt)
        print(root + '/' + protocols + '/' + file_txt)
        print(file_list)

        image_list = []
        label_list = []
        for i in range(len(file_list)):
            tmp_list = read_list(file_list[i])
            for name in tmp_list:
                name = revise_name(name)
                label_name = name.split("\\")[-2]
                img_name = '/'.join(name.split('\\'))
                #print(os.path.join(root, img_name))
                #print(int(label_name)%1000)
                image_list.append(os.path.join(root, img_name))
                label_list.append(int(label_name)%1000)
	
        self.image_list = image_list
        self.label_list = label_list
        self.class_nums =max(self.label_list)
        print(self.class_nums)

    def __getitem__(self, index):
        img_path = self.image_list[index]
        target = self.label_list[index]
        img = np.array(self.transform(Image.open(img_path)))

        if len(img.shape) == 2:
           img = np.stack([img] * 3, 2)
        #flip = np.random.choice(2)*2-1
        #img = img[:, ::flip, :]
        #img = (img - 127.5) / 128.0
        #img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        return img, target

    def __len__(self):
        return len(self.image_list)
if __name__ == '__main__':
    dat = CASIA_NIR_VIS('/root/NIR_VIS_Face_Recognition-master/dataset/CASIA_NIR_VIS_2.0/NIR-VIS-2.0')
    print(dat.__getitem__(100))
