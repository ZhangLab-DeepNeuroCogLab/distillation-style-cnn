import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import argparse
from PIL import Image
import random
import numpy as np
from operator import itemgetter
import copy
from functools import partial
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.transforms as transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy, InterpolationMode
import torch.backends.cudnn as cudnn

from models import *
from data.data_loader_imagenet import ExemplarDataset
from data.data_loader_imagenet import ImageNet100, ImageNet1K, ShapeBias, ImageNet100C, ImageNet100AR
from lib.util import *

# Seed
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

exemplar_sets = []
avg_acc = []
ft_avg_acc = []
ct_avg_acc = []

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # training hyperparameters
    parser.add_argument('--batch-size', type=int, default=128, help='batch_size')
    parser.add_argument('--num-workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--epochs1', type=int, default=70, help='number of training epochs')
    parser.add_argument('--epochs2', type=int, default=40, help='number of training epochs')
    parser.add_argument('--start-epoch', type=int, default=1, help='number of training epochs')
    parser.add_argument('--K', type=int, default=20, help='memory budget')
    parser.add_argument('--save-freq', type=int, default=1, help='memory budget')
    
    # incremental learning    
    parser.add_argument('--new-classes', type=int, default=10, help='number of classes in new task')
    parser.add_argument('--start-classes', type=int, default=50, help='number of classes in old task')

    #reply
    parser.add_argument('--is-reply', action='store_true', help='use reply')

    # optimization
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr-ft', type=float, default=0.01, help='learning rate for task-2 onwards')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--cosine', action='store_true', help='use cosine learning rate')

    # root folders
    # parser.add_argument('--train-data-root', type=str, default='../../data/imagenet', help='root directory of dataset')
    # parser.add_argument('--test-data-root', type=str, default='../../data/imagenet', help='root directory of dataset')
    # parser.add_argument('--style-data-root', type=str, default='../../data/style-imagenet', help='root directory of dataset')
    # parser.add_argument('--imagenetc-data-root', type=str, default='../../data/imagenet-c', help='root directory of dataset')
    # parser.add_argument('--imageneta-data-root', type=str, default='../../data/imagenet-a', help='root directory of dataset')
    # parser.add_argument('--imagenetr-data-root', type=str, default='../../data/imagenet-r', help='root directory of dataset')
    # parser.add_argument('--bias-data-root', type=str, default='../../data/shape_bias_100', help='root directory of dataset')
    parser.add_argument('--train-data-root', type=str, default='/data/temp_zenglin/data/imagenet', help='root directory of dataset')
    parser.add_argument('--test-data-root', type=str, default='/data/temp_zenglin/data/imagenet', help='root directory of dataset')
    parser.add_argument('--style-data-root', type=str, default='/data/temp_zenglin/data/data/style-imagenet',
                        help='root directory of dataset')
    parser.add_argument('--imagenetc-data-root', type=str, default='/data/temp_zenglin/data/imagenet-c',
                        help='root directory of dataset')
    parser.add_argument('--imageneta-data-root', type=str, default='/data/temp_zenglin/data/imagenet-a',
                        help='root directory of dataset')
    parser.add_argument('--imagenetr-data-root', type=str, default='/data/temp_zenglin/data/imagenet-r',
                        help='root directory of dataset')
    parser.add_argument('--bias-data-root', type=str, default='/data/temp_zenglin/data/shape_bias_100',
                        help='root directory of dataset')
    parser.add_argument('--output-root', type=str, default='./output', help='root directory for output')

    # dataset
    parser.add_argument('--dataset', type=str, default='imagenet100', choices=['imagenet100', 'imagenet'])

    # save and load
    parser.add_argument('--exp-name', type=str, default='img_aug', help='experiment name')
    parser.add_argument('--save', action='store_true', help='to save checkpoint')

    # loss function
    parser.add_argument('--pow', type=float, default=0.66, help='hyperparameter of adaptive weight')
    parser.add_argument('--lamda', type=float, default=100, help='weighting of classification and distillation')
    parser.add_argument('--const-lamda', action='store_true', help='use constant lamda value, default: adaptive weighting')
    parser.add_argument('--kd', action='store_true', help='use kd loss')
    parser.add_argument('--T', type=float, default=2, help='temperature scaling for KD')

    args = parser.parse_args()
    return args
        
def evaluate_acc(model, transform, test_classes):
    model.eval()

    valdir = os.path.join(args.test_data_root, 'val')
    if args.dataset == 'imagenet100':
        test_set = ImageNet100(valdir, train=False, classes=test_classes, transform=transform)
    elif args.dataset == 'imagenet':
        test_set = ImageNet1K(valdir, train=False, classes=test_classes, transform=transform)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=args.num_workers)
    
    total = 0.0
    correct = 0.0
    ft_correct = 0.0
    ft_total = 0.0
    ct_correct = 0.0
    ct_total = 0.0
    
    for j, (images, labels) in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            out = torch.softmax(model(images.cuda()), dim=1)
        
        _, preds = torch.max(out, dim=1, keepdim=False)
        preds = preds.cpu().numpy()
        labels = np.asarray([y.item() for y in labels])
        total += labels.size
        is_correct = np.asarray(preds == labels)
        correct += is_correct.sum()

        is_ft_classes = np.asarray([label in test_classes[:args.start_classes] for label in labels])
        ft_total += is_ft_classes.sum()
        ft_correct += (is_correct*is_ft_classes).sum()

        is_ct_classes = np.asarray([label in test_classes[-CLASS_NUM_IN_BATCH:] for label in labels])
        ct_total += is_ct_classes.sum()
        ct_correct += (is_correct*is_ct_classes).sum()

    test_acc = 100.0*correct/total
    ft_acc = 100.0*ft_correct/ft_total
    ct_acc = 100.0*ct_correct/ct_total
    # Test Accuracy
    print ('Acc : %.2f, FT Acc:%.2f, CT Acc:%.2f' % (test_acc,ft_acc,ct_acc))
    
    return test_acc, ft_acc, ct_acc

def evaluate_acc_ct_v1(model, transform, test_classes):
    model.eval()

    valdir = os.path.join(args.test_data_root, 'val')
    test_set = ImageNet100(valdir, train=False, classes=test_classes, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=args.num_workers)
    
    total = 0.0
    correct = 0.0
    
    for j, (images, labels) in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            out = torch.softmax(model(images.cuda()), dim=1)
        
        out = out[:,-CLASS_NUM_IN_BATCH:]
        _, preds = torch.max(out, dim=1, keepdim=False)
        preds = preds.cpu().numpy()
        labels = labels-CLASS_NUM_START
        labels = np.asarray([y.item() for y in labels])
        total += labels.size
        is_correct = np.asarray(preds == labels)
        correct += is_correct.sum()

    test_acc = 100.0*correct/total
    # Test Accuracy
    print ('CT Acc:%.2f' % test_acc)

def evaluate_acc_ct(model, transform, test_classes):
    model.eval()

    valdir = os.path.join(args.test_data_root, 'val')
    test_set = ImageNet100(valdir, train=False, classes=test_classes, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=args.num_workers)
    
    total = 0.0
    correct = 0.0
    
    for j, (images, labels) in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            out = torch.softmax(model(images.cuda()), dim=1)
        _, preds = torch.max(out, dim=1, keepdim=False)
        preds = preds.cpu().numpy()
        labels = np.asarray([y.item() for y in labels])
        total += labels.size
        is_correct = np.asarray(preds == labels)
        correct += is_correct.sum()

    test_acc = 100.0*correct/total
    # Test Accuracy
    print ('CT Acc:%.2f' % test_acc)
    return test_acc

def evaluate_acc_et(model, transform, test_classes):
    model.eval()

    valdir = os.path.join(args.test_data_root, 'val')
    test_set = ImageNet100(valdir, train=False, classes=test_classes, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=args.num_workers)
    
    total = 0.0
    correct = 0.0
    
    for j, (images, labels) in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            out = torch.softmax(model(images.cuda()), dim=1)
        
        out = out[:,test_classes]
        _, preds = torch.max(out, dim=1, keepdim=False)
        preds = preds.cpu().numpy()
        labels = labels-CLASS_NUM_START
        labels = np.asarray([y.item() for y in labels])
        total += labels.size
        is_correct = np.asarray(preds == labels)
        correct += is_correct.sum()

    test_acc = 100.0*correct/total
    # Test Accuracy
    print ('CT Acc:%.2f' % test_acc)

def evaluate_imagenetc(model, transform):
    model.eval()
    alexnet = AlexNet(100).cuda()
    alexnet.load_state_dict(torch.load('./models/alexnet_imagenet100_epoch70.pth'))
    alexnet.eval()

    ce_list = []
    ce_pairs = []
    for corr_type in ['gaussian_noise','shot_noise','impulse_noise',
                           'defocus_blur','glass_blur','motion_blur','zoom_blur',
                           'snow','frost','fog','brightness',
                           'contrast','elastic_transform','pixelate','jpeg_compression']:
        valdir = os.path.join(args.imagenetc_data_root, corr_type)
        test_set = ImageNet100C(valdir, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4)
        
        total = 0.0
        correct_net = 0.0
        correct_alexnet = 0.0
        
        for (images, labels) in tqdm(test_loader):
            with torch.no_grad():
                out_net = torch.softmax(model(images.cuda()), dim=1)
                out_alexnet = torch.softmax(alexnet(images.cuda()), dim=1)
            _, net_preds = torch.max(out_net, dim=1, keepdim=False)
            net_preds = net_preds.cpu().numpy()

            _, out_alexnet = torch.max(out_alexnet, dim=1, keepdim=False)
            out_alexnet = out_alexnet.cpu().numpy()

            labels = np.asarray([y.item() for y in labels])
            total += labels.size

            is_correct_net = np.asarray(net_preds == labels)
            correct_net += is_correct_net.sum()

            is_correct_alexnet = np.asarray(out_alexnet == labels)
            correct_alexnet += is_correct_alexnet.sum()

        test_err_net = 1-correct_net/total
        test_err_alexnet = 1-correct_alexnet/total
        print(corr_type,test_err_net*100,test_err_alexnet*100)
        ce = 100.0*test_err_net/test_err_alexnet
        ce_list.append(ce)
        ce_pairs.append((corr_type,ce))
    mce = np.array(ce_list)
    mce = np.mean(mce)
    print ('ImagenetC mce: %.2f' % mce)
    print (ce_pairs)


def evaluate_imagenetr(model, transform):
    model.eval()
    test_set = ImageNet100AR(args.imagenetr_data_root, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=args.num_workers)
    
    total = 0.0
    correct = 0.0
    
    for j, (images, labels) in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            out = torch.softmax(model(images.cuda()), dim=1)
        _, preds = torch.max(out, dim=1, keepdim=False)
        preds = preds.cpu().numpy()
        labels = np.asarray([y.item() for y in labels])
        total += labels.size
        is_correct = np.asarray(preds == labels)
        correct += is_correct.sum()

    test_acc = 100.0*correct/total
    # Test Accuracy
    print ('ImagenetR Accuracy : %.2f' % test_acc)

if __name__ == '__main__':
    args = parse_option()
    print (args)

    # if not os.path.exists(os.path.join(args.output_root, "checkpoints/imagenet100/")):
    #     os.makedirs(os.path.join(args.output_root, "checkpoints/imagenet100/"))

    #  parameters
    if args.dataset == 'imagenet100':
        TOTAL_CLASS_NUM = 100
    elif args.dataset == 'imagenet':
        TOTAL_CLASS_NUM = 1000

    # test-time augmentation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    class_index = [i for i in range(0, TOTAL_CLASS_NUM)]
    net = resnet18_imagenet(num_classes=args.start_classes).cuda()

    CLASS_NUM_IN_BATCH = args.start_classes
    save_path = os.path.join(args.output_root, "checkpoints/imagenet100",args.exp_name+'_0.pth')
    net.load_state_dict(torch.load(save_path))
    net = net.cuda()
    CLASS_NUM_START = 0
    #evaluate_acc(model=net, transform=transform_test, test_classes=class_index[:args.start_classes])
    #evaluate_acc_ct(model=net, transform=transform_test, test_classes=class_index[:CLASS_NUM_IN_BATCH])
    acc_ii=[]
    acc_ti=[]
    cls_list = [0]+[a for a in range(args.start_classes, TOTAL_CLASS_NUM, args.new_classes)]
    for i in cls_list:
        CLASS_NUM_START = i
        if i == args.start_classes:
            CLASS_NUM_IN_BATCH = args.new_classes
            net.change_output_dim(new_dim=i+args.new_classes)
        if i > args.start_classes:
            net.change_output_dim(new_dim=i+args.new_classes, second_iter=True)

        save_path = os.path.join(args.output_root, "checkpoints/imagenet100","\'"+args.exp_name+"\'"+'_%d.pth'%i)
        net.load_state_dict(torch.load(save_path))
        net = net.cuda()
        #evaluate_acc(model=net, transform=transform_test, test_classes=class_index[:i+CLASS_NUM_IN_BATCH])
        acc_ct=evaluate_acc_ct(model=net, transform=transform_test, test_classes=class_index[i:i+CLASS_NUM_IN_BATCH])
        acc_ii.append(acc_ct)

    CLASS_NUM_IN_BATCH=args.start_classes
    cls_list = [0]+[a for a in range(args.start_classes, TOTAL_CLASS_NUM, args.new_classes)]
    for i in cls_list:
        CLASS_NUM_START = i
        if i == args.start_classes:
            CLASS_NUM_IN_BATCH = args.new_classes
        acc_ct=evaluate_acc_ct(model=net, transform=transform_test, test_classes=class_index[i:i+CLASS_NUM_IN_BATCH])
        acc_ti.append(acc_ct)

    acc_ii = np.asarray(acc_ii)
    acc_ti = np.asarray(acc_ti)
    print ('Avg accuracy: ', sum(acc_ti[:-1]-acc_ii[:-1])/len(acc_ti[:-1]))
    #evaluate_shape_bias(model=net, transform=transform_test,cls_names=cls_names)
    #evaluate_sin_acc(model=net, transform=transform_test, test_classes=class_index)
    #evaluate_imagenetc(model=net, transform=transform_test)
    #evaluate_imageneta(model=net, transform=transform_test)
    #evaluate_imagenetr(model=net, transform=transform_test)