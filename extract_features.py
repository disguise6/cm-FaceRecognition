'''
    implement the feature extractions for light CNN
    @author: Alfred Xiang Wu
    @date: 2017.07.04
'''
### revised by Zhenggang Li 2018.12.10

from __future__ import print_function
import argparse
import os
import shutil
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from core import IDR
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
import numpy as np
import cv2
from load_imglist import ImageList

# Use default value
parser = argparse.ArgumentParser(description='PyTorch ImageNet Feature Extracting')
parser.add_argument('--arch', '-a', metavar='ARCH', default='LightCNN')
parser.add_argument('--cuda', '-c', default=True)
parser.add_argument('--model', default='IDRnet', type=str, metavar='Model',
                    help='model type: LightCNN-9, LightCNN-29')
parser.add_argument('--num_classes', default=839, type=int,
                    metavar='N', help='mini-batch size (default: 713)')

## input if necessary
parser.add_argument('--root_path', default='', type=str, metavar='PATH', 
                    help='root path of face images (default: none).')
parser.add_argument('--resume', default='model/tmp/lightCNN_60_checkpoint.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--img_list', default='', type=str, metavar='PATH', 
                    help='list of face images for feature extraction (default: none).')
parser.add_argument('--protocols', default='', type=str, metavar='PATH', 
                    help='list of protocols (default: none).')
# parser.add_argument('--save_path', default='', type=str, metavar='PATH', 
#                     help='save root path for features of face images.')

postfix = None

def main():
    stat = []
    print()
    print('---------------------------------')

    avg_r_a, std_r_a, avg_v_a, std_v_a = excute()
    # print(avg_rank1_acc, avg_vr_acc)
    stat.append([avg_r_a, std_r_a, avg_v_a, std_v_a])
    print(stat)
    exit()

def excute():
    global args
    args = parser.parse_args()

    # print(args)
    # exit()
    if args.root_path == '':
        args.root_path = r'/root/NIR_VIS_Face_Recognition-master/dataset/CASIA_NIR_VIS_2.0/NIR-VIS-2.0'
    if args.resume == '':
        args.resume = 'LightCNN_9Layers_checkpoint.pth.tar'
    if args.protocols == '':
        args.protocols = 'MyProtocols'
    
    # create Light CNN for face recognition
    if args.model == 'IDRnet':
        model = IDR.IDRnet(num_classes=args.num_classes)
    else:
        print('Error model type\n')

    model.eval()
    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    gallery_file_list = 'mix_vis_gallery.txt'
    probe_file_list = 'mix_nir_probe.txt'
    import glob2
    
    gallery_file_list = glob2.glob(args.root_path + '/' + args.protocols + '/' + gallery_file_list)
    probe_file_list = glob2.glob(args.root_path + '/' + args.protocols + '/' + probe_file_list)
    # remove *_dev.txt file in both list
    print(gallery_file_list)
    
    avg_r_a, std_r_a, avg_v_a, std_v_a = load(model, args.root_path, gallery_file_list, probe_file_list)
    return avg_r_a, std_r_a, avg_v_a, std_v_a
    
def load(model, root_path, gallery_file_list, probe_file_list):
    cosine_similarity_score = []
    rank1_acc = []
    vr_acc = []
    g_count = 0
    # read sub_experiment from gallery file list 0-10
    for i in range(len(gallery_file_list)):
        g_count += 1
        
        transform = transforms.Compose([
            transforms.Grayscale(1),
            transforms.ToTensor(),
        ])
        count     = 0
        input     = torch.zeros(1, 1, 128, 128)
        ccount    = 0
        simi_dict = {}
        probe_features = []
        gallery_features = []
        probe_names = []
        gallery_names = []
        probe_img_name_list = []
        gallery_img_name_list = []

        # read images in each sub_experiment
        # in the first time read images from probe image file 1
        print('==> extract feature from the image')

        probe_img_list = read_list(probe_file_list[i])
        print('===> probe image list')
        for probe_img_name in probe_img_list:
            # break
            probe_img_name = revise_name(probe_img_name)
            #print("probe_img_name:",probe_img_name)
            
            
            start = time.time()
            count = count + 1
            label = 0
            simi_dict[count] = []
            probe_img_feature, target = feature_extract(probe_img_name, input, transform, model, root_path)
            # save_feature(root_path, probe_img_name, probe_img_feature)
            
            # features.data.cpu().numpy()[0]
            end = time.time() - start
            # print("{}({}/{}). Time: {:.4f}".format(os.path.join(root_path, probe_img_name), count, len(probe_img_list), end))
            # if count == 10:
            #     break

            probe_features.append(probe_img_feature)
            probe_names.append(target)
            probe_img_name_list.append(probe_img_name)


        gallery_img_list  = read_list(gallery_file_list[i])
        print('===> gallery image list')
        for gallery_img_name in gallery_img_list:
            # break
            gallery_img_name = revise_name(gallery_img_name)
            #print("gallery_img_name:",gallery_img_name)
            start = time.time()
            ccount = ccount + 1
            gallery_img_feature, target = feature_extract(gallery_img_name, input, transform, model, root_path)
            # print(gallery_img_feature.shape)
            # save_feature(root_path, gallery_img_name, gallery_img_feature)
            # exit()
            
            
            end = time.time() - start
            # print("{}({}/{}). Time: {:.4f}".format(os.path.join(root_path, gallery_img_name), ccount, len(gallery_img_list), end))
            # continue

            gallery_features.append(gallery_img_feature)
            gallery_names.append(target)
            gallery_img_name_list.append(gallery_img_name)


        # 计算特征向量相似度
        probe_features = np.array(probe_features)
        gallery_features = np.array(gallery_features)

        # print('probe_features.shape=', probe_features.shape)
        # print('gallery_features.shape=', gallery_features.shape)
        
        from sklearn.metrics.pairwise import cosine_similarity
        
        score = cosine_similarity(gallery_features, probe_features).T
        # print('score.shape= ', score.shape)
        # exit()
        r_acc, tpr = compute_metric(score, probe_names, gallery_names, g_count, probe_img_name_list, gallery_img_name_list)
        # print('score={}, probe_names={}, gallery_names={}'.format(score, probe_names, gallery_names))
        rank1_acc.append(r_acc)
        vr_acc.append(tpr)
    
    # print('over')
    # exit()


    avg_r_a = np.mean(np.array(rank1_acc))
    std_r_a = np.std(np.array(rank1_acc))
    avg_v_a = np.mean(np.array(vr_acc))
    std_v_a = np.std(np.array(vr_acc))
    # print(avg)
    # avg_rank1_acc = sum(rank1_acc)/(len(rank1_acc) + 1e-5)
    # avg_vr_acc = sum(vr_acc)/(len(vr_acc) + 1e-5)
    print()
    print('=====================================================')
    print('Final Rank1 accuracy is', avg_r_a * 100, "% +", std_r_a)
    print('Final VR@FAR=0.1% accuracy is', avg_v_a * 100, "% +", std_v_a)
    print('=====================================================')
    print()
    return avg_r_a, std_r_a, avg_v_a, std_v_a

def revise_name(probe_img_name):
    # print(probe_img_name, type(probe_img_name))
    suffix = probe_img_name.split('.')
    if suffix[-1] != 'bmp':
        suffix[-1] = 'bmp'
    
    probe_img_name = '.'.join(suffix)
    revise_name = probe_img_name.split('\\')
    # print(revise_name)
    # # use '_128x128' when evaluate cropped image provided by dataset
    # revise_name[1] += '_128x128' 
    # # use '_crop' when evaluate cropped image provided by zhenggang, scale = 48/100 
    # # use '_80' when evaluate cropped image provided by zhenggang, scale = 48/80
    # # use '_80' when evaluate cropped image provided by zhenggang, scale = 48/110
    #global postfix   #sgong
    # revise_name[1] += '_train'
    #revise_name[1] += '_crop' + str(postfix) 
    #revise_name[1] += '_128x128'  #sgong 
    # print(revise_name)
    # exit()
    temp = ""
    for i in range(len(revise_name)):
        temp = temp + revise_name[i]
        if i != len(revise_name) -1:
            temp += '\\'
    return temp

def compute_metric(score, probe_names, gallery_names, g_count, probe_img_list, gallery_img_list):
    # print('score.shape =', score.shape)
    # print('probe_names =', np.array(probe_names).shape)
    # print('gallery_names =', np.array(gallery_names).shape)
    print('===> compute metrics')
    # print(probe_names[1], type(probe_names[1]))
    # exit()
    label = np.zeros_like(score)
    maxIndex = np.argmax(score, axis=1)
    # print('len = ', len(maxIndex))
    count = 0
    for i in range(len(maxIndex)):
        probe_names_repeat = np.repeat([probe_names[i]], len(gallery_names), axis=0).T
        # compare two string list
        result = np.equal(probe_names_repeat, gallery_names) * 1
        # result = np.core.defchararray.equal(probe_names_repeat, gallery_names) * 1
        # find the index of image in the gallery that has the same name as probe image
        # print(result)
        # print('++++++++++++++++++++++++++++++++=')
        index = np.nonzero(result==1)
        
        # if i == 10:
        #     exit()
        if len(index[0]) != 1:
            print('more than one identity name in gallery is same as probe image name')
            ind = index[0]
        else:
            label[i][index[0][0]] = 1
            ind = index[0][0]
        
        # find the max similarty score in gallery has the same name as probe image
        if np.equal(int(probe_names[i]), int(gallery_names[maxIndex[i]])):
            count += 1
        else:
            pass
            # print(probe_img_list[i], gallery_img_list[ind])
            
        # flag = np.equal(probe_names[i], gallery_names[i])

        # labelOfmaxIndex.append(label[0, i])
    # print('count = ', count)
    r_acc = count/(len(probe_names)+1e-5)

    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(label.flatten(), score.flatten())
    
    print("In sub_experiment", g_count, 'count of true label :', count)
    print('rank1 accuracy =', r_acc)
    print('VR@FAR=0.1% accuracy =', tpr[fpr <= 0.001][-1])
    
    # plot_roc(fpr, tpr, thresholds, g_count)
    return r_acc, tpr[fpr <= 0.001][-1]

def feature_extract(img_name, input, transform, model, root_path):
    # 得到标签
    target = img_name.split("\\")[-2]
    type_str = img_name.split("\\")[1][:3]
    #print('type_str:{}'.format(type_str))
    idt = 0
    if type_str == 'VIS':
        idt = 1
    # print(target)
    img_name = '/'.join(img_name.split('\\'))
    # print(img_name)
    path = root_path
    # print(os.path.join(path, img_name))

    img = transform(Image.open(os.path.join(path, img_name)))
    
    if img is None:
        print('image not found')

    input[0, :, :, :] = img
    #print(input.shape)

    input = torch.tensor(input).cuda()
    #print(type(input))
    idt = torch.tensor(idt).reshape((1, )).cuda()
    #print(idt.size())
    with torch.no_grad():
        input_var = torch.autograd.Variable(input)
        idt = torch.autograd.Variable(idt)

    _, features = model(input_var, idt)
    # the type of features is a tensor.cuda, which means your data in GPU cache
    # in order to calculate it with numpy array in cpu, needs to replicate your data to cpu first
    # features.data.cpu().numpy()[0]
    return features.data.cpu().numpy()[0], int(target)

def read_list(list_path):
    img_list = []
    with open(list_path, 'r') as f:
        for line in f.readlines()[0:]:
            img_path = line.strip().split()
            img_list.append(img_path[0])
    print('There are {} images..'.format(len(img_list)))
    return img_list

def save_feature(save_path, img_name, features):
    # print(save_path, img_name)
    i_n = img_name.split('\\')
    # print(i_n)
    # i_n[1] = i_n[1] +'_feat'
    if 'NIR' in i_n[1]:
        # print(i_n[1])
        i_n[1] = 'NIR_crop6_feat'
    else:
        i_n[1] = 'VIS_crop6_feat'
    # folder_name = '/'.join(i_n[0:-1])
    # print(folder_name)
    # exit()

    i_n[-1] = i_n[-1].split('.')[0]
    img_name = '/'.join(i_n)

    # print(img_name)
    
    img_path = os.path.join(save_path, img_name)
    img_dir  = os.path.dirname(img_path) + '/';
    # print(img_dir)

    # exit()
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    # fname = os.path.splitext(img_path)[0]
    fname = img_path + '.npy'
    
    np.save(fname, features)
    

if __name__ == '__main__':
    main()
