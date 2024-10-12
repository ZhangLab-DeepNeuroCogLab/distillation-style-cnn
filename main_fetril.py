import datetime
import json
import logging
import os
import sys
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
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models import *
from data.data_loader_imagenet import ExemplarDataset
from data.data_loader_imagenet import ImageNet100, ImageNet1K, ShapeBias, ImageNet100C, ImageNet100AR
from lib.util import *

# Seed
seed = random.randint(1, 1000)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
sys.excepthook = handle_uncaught_exception

exemplar_sets = []
avg_acc = []
ft_avg_acc = []
MEAN = torch.tensor([0.485, 0.456, 0.406]).cuda()
STD = torch.tensor([0.229, 0.224, 0.225]).cuda()


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # training hyperparameters
    # parser.add_argument('--local_rank', type=int, default=os.getenv('LOCAL_RANK', -1), help='local_rank')
    parser.add_argument('--batch-size', type=int, default=256, help='batch_size')
    parser.add_argument('--num-workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--epochs1', type=int, default=70, help='number of training epochs')
    parser.add_argument('--epochs2', type=int, default=40, help='number of training epochs')
    parser.add_argument('--start-epoch', type=int, default=1, help='number of training epochs')
    parser.add_argument('--K', type=int, default=20, help='memory budget')
    parser.add_argument('--save-freq', type=int, default=1, help='save model frequency')

    # incremental learning    
    parser.add_argument('--new-classes', type=int, default=10, help='number of classes in new task')
    parser.add_argument('--start-classes', type=int, default=50, help='number of classes in old task')

    # replay
    parser.add_argument('--is-reply', action='store_true', help='use reply')

    # optimization
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr-ft', type=float, default=0.01, help='learning rate for task-2 onwards')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--cosine', action='store_true', help='use cosine learning rate')

    # root folders
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
    parser.add_argument('--const-lamda', action='store_true',
                        help='use constant lamda value, default: adaptive weighting')
    parser.add_argument('--kd', action='store_true', help='use kd loss')
    parser.add_argument('--T', type=float, default=2, help='temperature scaling for KD')
    # new added arguments for running
    parser.add_argument('--no_STCR', action='store_true', help='whether to use STCR')
    parser.add_argument('--no_distill', action='store_true', help='whether to use distillation loss')
    parser.add_argument('--style_type', type=int, default=0, help='style transfer method, 0 as adain, 1 as tradition')


    args = parser.parse_args()
    return args


def train(model, old_model, train_loader, style_transfer):
    is_reply = args.is_reply
    T = args.T
    model.cuda()

    criterion_ce = nn.CrossEntropyLoss(ignore_index=-1)

    if len(test_classes) // CLASS_NUM_IN_BATCH > 1:
        epoch = args.epochs2
        lr = args.lr_ft
    else:
        epoch = args.epochs1
        lr = args.lr

    if args.start_epoch == 1:
        logging.info('setting optimizer and scheduler.................')
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=args.weight_decay)
        if len(test_classes) // CLASS_NUM_IN_BATCH == 1:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 35], gamma=0.1)

    if is_reply and len(test_classes) // CLASS_NUM_IN_BATCH > 1:
        exemplar_set = ExemplarDataset(exemplar_sets, transform=transform_train, transform_style=transform_style)
        exemplar_loader = torch.utils.data.DataLoader(exemplar_set, batch_size=args.batch_size,
                                                      num_workers=args.num_workers, shuffle=True, drop_last=True)
        exemplar_loader_iter = iter(exemplar_loader)

    old_model.eval()

    for epoch_index in range(args.start_epoch, epoch + 1):
        sum_loss = 0
        sum_dist_bn_loss = 0
        sum_dist_loss = 0
        sum_cls_new_loss = 0
        sum_cls_old_loss = 0

        model.train()
        old_model.eval()

        for (x, x_style, target) in tqdm(train_loader):
            x, x_style, target = x.cuda(), x_style.cuda(), target.cuda()
            if is_reply and len(test_classes) // CLASS_NUM_IN_BATCH > 1:
                try:
                        batch_ex = next(exemplar_loader_iter)
                except:
                    exemplar_loader_iter = iter(exemplar_loader)
                    batch_ex = next(exemplar_loader_iter)
                x_old, x_old_style, target_old = batch_ex  #
                x_old, x_old_style, target_old = x_old.cuda(), x_old_style.cuda(), target_old.cuda()
                random_index = torch.randperm(x_old.shape[0])

            if not args.no_STCR: 
                if is_reply and len(test_classes) // CLASS_NUM_IN_BATCH > 1:
                    x_s = style_transfer(x, x_old_style[random_index[:x.shape[0]]], alpha=1)
                else:
                    random_index = torch.randperm(x.shape[0])
                    x_s = style_transfer(x, x_style[random_index], alpha=1)
                    
                # xs = (xs - MEAN[:, None, None]) / STD[:, None, None]
                x = torch.cat((x, x_s), 0)
                x = (x - MEAN[:, None, None]) / STD[:, None, None]
                target = torch.cat((target, target), 0)

            # Classification Loss: New task          
            targets = target - len(test_classes) + CLASS_NUM_IN_BATCH
            logits = model(x)
            cls_loss_new = criterion_ce(logits[:, -CLASS_NUM_IN_BATCH:], targets)
            loss = cls_loss_new
            sum_cls_new_loss += cls_loss_new.item()
            if not args.no_distill:

                # batch distillation
                N = x.shape[0]
                logits_dist_bn = logits[:, -CLASS_NUM_IN_BATCH:]
                logits_ens = logits_dist_bn.clone().detach()
                logits_in = logits_ens[:N // 2, -CLASS_NUM_IN_BATCH:]
                logits_sin = logits_ens[N // 2:, -CLASS_NUM_IN_BATCH:]
                logits_ens_temp = (logits_in + logits_sin) / 2
                logits_ens[:N // 2, -CLASS_NUM_IN_BATCH:] = logits_ens_temp
                logits_ens[N // 2:, -CLASS_NUM_IN_BATCH:] = logits_ens_temp
                dist_bn_loss_new = nn.KLDivLoss()(F.log_softmax(logits_dist_bn / T, dim=1),
                                                  F.softmax(logits_ens / T, dim=1)) * (T * T)
                loss += 0.1 * dist_bn_loss_new
                sum_dist_bn_loss += dist_bn_loss_new.item()

            # use fixed lamda value or adaptive weighting 

            if args.kd and len(test_classes) // CLASS_NUM_IN_BATCH > 1:
                if args.const_lamda:
                    factor = args.lamda
                else:
                    factor = ((len(test_classes) / CLASS_NUM_IN_BATCH) ** (args.pow)) * args.lamda

                with torch.no_grad():
                    dist_target = old_model(x)
                logits_dist = logits[:, :-CLASS_NUM_IN_BATCH]
                dist_loss_new = nn.KLDivLoss()(F.log_softmax(logits_dist / T, dim=1),
                                               F.softmax(dist_target / T, dim=1)) * (T * T)
                loss += factor * dist_loss_new

            if is_reply and len(test_classes) // CLASS_NUM_IN_BATCH > 1:
                """
                x_s_old = style_transfer(x_old, None, alpha=1)
                x_old = torch.cat((x_old, x_s_old), 0)
                target_old = torch.cat((target_old, target_old), 0)#"""

                x_old = (x_old - MEAN[:, None, None]) / STD[:, None, None]
                logits_old = model(x_old)
                old_classes = len(test_classes) - CLASS_NUM_IN_BATCH
                cls_loss_old = criterion_ce(logits_old, target_old)
                loss += cls_loss_old
                sum_cls_old_loss += cls_loss_old.item()

                # KD loss using exemplars
                if args.kd:
                    with torch.no_grad():
                        dist_target_old = old_model(x_old)
                    logits_dist_old = logits_old[:, :-CLASS_NUM_IN_BATCH]
                    dist_loss_old = nn.KLDivLoss()(F.log_softmax(logits_dist_old / T, dim=1),
                                                   F.softmax(dist_target_old / T, dim=1)) * (T * T)  # best model
                    loss += factor * dist_loss_old

                    dist_loss = dist_loss_old + dist_loss_new
                    sum_dist_loss += dist_loss.item()

            sum_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_len = len(trainLoader)
        logging.info(
    '==>>> epoch: {}, loss: {:.3f}, cls_new_loss: {:.3f}, cls_old_loss: {:.3f}, dist_loss: {:.3f}, dist_bn_loss: {:.3f}'.format(
        epoch_index, 
        sum_loss / train_len if train_len != 0 else float('nan'), 
        sum_cls_new_loss / train_len if train_len != 0 else float('nan'), 
        sum_cls_old_loss / train_len if train_len != 0 else float('nan'),
        sum_dist_loss / train_len if train_len != 0 else float('nan'), 
        sum_dist_bn_loss / train_len if train_len != 0 else float('nan')
    )
)

        if epoch_index % 10 == 0:
            test_acc, ft_acc = evaluate_acc(model=model, transform=transform_test, test_classes=test_classes)

        scheduler.step()


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
        ft_correct += (is_correct * is_ft_classes).sum()

    test_acc = 100.0 * correct / total
    ft_acc = 100.0 * ft_correct / ft_total
    # Test Accuracy
    logging.info('Test Accuracy : %.2f, First Task Accuracy:%.2f' % (test_acc, ft_acc))

    return test_acc, ft_acc


def evaluate_sin_acc(model, transform, test_classes):
    model.eval()

    valdir = os.path.join(args.style_data_root, 'val')
    if args.dataset == 'imagenet100':
        test_set = ImageNet100(valdir, train=False, classes=test_classes, transform=transform)
    elif args.dataset == 'imagenet':
        test_set = ImageNet1K(valdir, train=False, classes=test_classes, transform=transform)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4)

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

    test_acc = 100.0 * correct / total
    logging.info('Test sin Accuracy : %.2f' % test_acc)


def evaluate_imagenetc(model, transform):
    model.eval()
    alexnet = AlexNet().cuda()
    alexnet.load_state_dict(torch.load('./models/alexnet_imagenet100_epoch70.pth'))
    alexnet.eval()

    ce_list = []
    ce_pairs = []
    for corr_type in ['gaussian_noise', 'shot_noise', 'impulse_noise',
                      'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                      'snow', 'frost', 'fog', 'brightness',
                      'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']:
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

        test_err_net = 1 - correct_net / total
        test_err_alexnet = 1 - correct_alexnet / total
        logging.info(f"{corr_type}, {test_err_net * 100}, {test_err_alexnet * 100}")
        ce = 100.0 * test_err_net / test_err_alexnet
        ce_list.append(ce)
        ce_pairs.append((corr_type, ce))
    mce = np.array(ce_list)
    mce = np.mean(mce)
    logging.info('ImagenetC mce: %.2f' % mce)
    logging.info(str(ce_pairs))
    return mce


def evaluate_imageneta(model, transform):
    model.eval()
    test_set = ImageNet100AR(args.imageneta_data_root, transform=transform)
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

    test_acc = 100.0 * correct / total
    # Test Accuracy
    logging.info('ImagenetA Accuracy : %.2f' % test_acc)
    return test_acc


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

    test_acc = 100.0 * correct / total
    # Test Accuracy
    logging.info('ImagenetR Accuracy : %.2f' % test_acc)


def evaluate_shape_bias(model, transform, cls_names):
    model.eval()
    test_set = ShapeBias(args.bias_data_root, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=args.num_workers)

    texture_corrext = 0.0
    shape_correct = 0.0

    for (images, labels) in tqdm(test_loader):
        with torch.no_grad():
            out = torch.softmax(model(images.cuda()), dim=1)

        probabilities = out[0].cpu().numpy()
        decision = probabilities_to_decision(probabilities, cls_names)
        shape, texture = labels[0].split('-')
        if decision in shape and decision in texture:
            pass
        else:
            if decision in shape:
                shape_correct += 1
            elif decision in texture:
                texture_corrext += 1

    shape_bias = 100.0 * shape_correct / (shape_correct + texture_corrext)
    shape_match = 100.0 * shape_correct / len(test_loader)
    texture_match = 100.0 * texture_corrext / len(test_loader)

    logging.info('shape_bias: %0.2f, shape_match: %0.2f, texture_match: %0.2f ' % (shape_bias, shape_match, texture_match))


# Construct an exemplar set for image set
def icarl_construct_exemplar_set(model, images_path, m, transform):
    """
    exemplar_set = images_path[:m]
    exemplar_sets.append(exemplar_set)
    print ('exemplar set shape: ', len(exemplar_set))#"""

    # """
    model.eval()
    # Compute and cache features for each example
    features = []
    with torch.no_grad():
        x_batch = []
        for id, img_path in enumerate(images_path):
            if (id + 1) % args.batch_size > 0 and (id + 1) < len(images_path):
                img = np.array(cv2.imread(img_path))
                x = transform(Image.fromarray(img))
                x_batch.append(x)
            else:
                if len(x_batch) > 8:
                    x_batch = torch.stack(x_batch)
                    x_batch = x_batch.cuda()  # Variable()
                    feat_batch = model.forward(x_batch, feat=True).data.cpu().numpy()
                    for i_batch in range(feat_batch.shape[0]):
                        feat = feat_batch[i_batch]
                        feat = feat / np.linalg.norm(feat)  # Normalize
                        features.append(feat)
                x_batch = []

        features = np.array(features)
        class_mean = np.mean(features, axis=0)
        class_mean = class_mean / np.linalg.norm(class_mean)  # Normalize

        exemplar_set = []
        exemplar_features = []  # list of Variables of shape (feature_size,)
        exemplar_dist = []
        for k in range(int(m)):
            S = np.sum(exemplar_features, axis=0)
            phi = features
            mu = class_mean
            mu_p = 1.0 / (k + 1) * (phi + S)
            mu_p = mu_p / np.linalg.norm(mu_p)
            dist = np.sqrt(np.sum((mu - mu_p) ** 2, axis=1))

            i = np.random.randint(0, features.shape[0])

            exemplar_dist.append(dist[i])
            exemplar_set.append(images_path[i])
            exemplar_features.append(features[i])
            features[i, :] = 0.0

        # random exemplar selection
        exemplar_dist = np.array(exemplar_dist)
        # exemplar_set = np.array(exemplar_set)
        ind = exemplar_dist.argsort()
        exemplar_set = itemgetter(*ind)(exemplar_set)  # exemplar_set[ind]

        exemplar_sets.append(exemplar_set)
    logging.info('exemplar set shape: {}.'.format(len(exemplar_set)))  # """


if __name__ == '__main__':
    commands = ' '.join(sys.argv)
    args = parse_option()
    if args.dataset == 'imagenet100':
        TOTAL_CLASS_NUM = 100
    elif args.dataset == 'imagenet':
        TOTAL_CLASS_NUM = 1000
    
    now = datetime.datetime.now()
    time_label = now.strftime("%m%d%H%M")
    exp_name = f"time_{time_label}_start_{args.start_classes}_new_{args.new_classes}_cosine_{args.cosine}_kd_{args.kd}"
    save_path = os.path.join(args.output_root, f"checkpoints/{args.dataset}/", exp_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(save_path + '/log.txt'),
        logging.StreamHandler()
    ]
    )
    logging.info(args)


    if not os.path.exists(os.path.join(args.output_root, "checkpoints/imagenet100/")):
        os.makedirs(os.path.join(args.output_root, "checkpoints/imagenet100/"))

    #  parameters


    CLASS_NUM_IN_BATCH = args.start_classes
    TOTAL_CLASS_BATCH_NUM = TOTAL_CLASS_NUM // CLASS_NUM_IN_BATCH
    is_reply = args.is_reply

    # default augmentation
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    # style augmentation
    color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
    transform_style = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur(kernel_size=int(0.1 * 224))], p=0.5),
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # transforms.RandomApply([GaussianNoise()], p=0.5)
    ])

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
    net = resnet18_imagenet(num_classes=CLASS_NUM_IN_BATCH).cuda()
    if args.style_type==0:
        style_transfer = StyleTransfer()
    else:
        style_transfer = style_transfer2

    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    logging.info('number of trainable parameters: {}'.format(params))

    old_net = copy.deepcopy(net)
    old_net.cuda()

    cls_list = [0] + [a for a in range(args.start_classes, TOTAL_CLASS_NUM, args.new_classes)]

    for i in cls_list:

        if i == args.start_classes:
            CLASS_NUM_IN_BATCH = args.new_classes

        logging.info("==> Current Class: {}".format(class_index[i:i + CLASS_NUM_IN_BATCH]))
        logging.info('==> Building model..')

        if i == args.start_classes:
            net.change_output_dim(new_dim=i + CLASS_NUM_IN_BATCH)
        if i > args.start_classes:
            net.change_output_dim(new_dim=i + CLASS_NUM_IN_BATCH, second_iter=True)

        traindir = os.path.join(args.train_data_root, 'train')

        if args.dataset == 'imagenet100':
            train_set = ImageNet100(traindir, train=True, classes=class_index[i:i + CLASS_NUM_IN_BATCH],
                                    transform=transform_train, transform_style=transform_style)
        elif args.dataset == 'imagenet':
            train_set = ImageNet1K(traindir, train=True, classes=class_index[i:i + CLASS_NUM_IN_BATCH],
                                   transform=transform_train)

        cls_names = train_set.get_cls_names()
        trainLoader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size // 2,
                                                  num_workers=args.num_workers, shuffle=True, drop_last=True)

        train_classes = class_index[i:i + CLASS_NUM_IN_BATCH]
        test_classes = class_index[:i + CLASS_NUM_IN_BATCH]

        logging.info(train_classes)
        logging.info(test_classes)
        #icarl_construct_exemplar_set using image_path for this experience to build exemplar set
        if is_reply:
            m = args.K
            for y in range(i, i + CLASS_NUM_IN_BATCH):
                logging.info("Constructing exemplar set for class-%d..." % (class_index[y]))
                images_path = train_set.get_image_class(y)
                icarl_construct_exemplar_set(net, images_path, m, transform_test)
                logging.info("Done")

        """
        if i==0:
            save_path = os.path.join(args.output_root, "checkpoints/imagenet100",args.exp_name+'_0.pth')
            net.load_state_dict(torch.load(save_path))
            net.train()
        else:#"""
        net.train()
        train(model=net, old_model=old_net, train_loader=trainLoader, style_transfer=style_transfer)
        old_net = copy.deepcopy(net)
        old_net.cuda()

        model_save_path = os.path.join(save_path, "checkpoints/")
        if not os.path.exists(os.path.dirname(model_save_path)):
            os.makedirs(os.path.dirname(model_save_path))
        model_save_path = os.path.join(model_save_path, args.exp_name + '_%d.pth' % i)
        torch.save(net.state_dict(), model_save_path)

        # ---------------------- Evaluation ----------------------------------
        test_acc, ft_acc = evaluate_acc(model=net, transform=transform_test, test_classes=test_classes)
        avg_acc.append(test_acc)
        ft_avg_acc.append(ft_acc)

    logging.info('ft_avg_acc_list:%s' % ft_avg_acc)
    logging.info('Ft Avg accuracy: {}'.format(sum(ft_avg_acc) / len(ft_avg_acc)))

    logging.info('avg_acc_list:%s' % avg_acc)
    logging.info('Avg accuracy: {}'.format(sum(avg_acc) / len(avg_acc)))

    r_c = evaluate_imagenetc(model=net, transform=transform_test)
    r_r = evaluate_imagenetr(model=net, transform=transform_test)
    examplar_save_path = os.path.join(save_path, "checkpoints/", args.exp_name + '.txt')
    with open(examplar_save_path, 'w') as f:
        f.write('%s' % exemplar_sets)
        f.close()
    # Check if output.json exists, if not create an empty dictionary
    if os.path.exists(os.path.join(save_path, "checkpoints/", 'output.json')):
        with open(os.path.join(save_path, "checkpoints/",  'output.json'), 'r') as f:
            json_data = json.load(f)
    else:
        json_data = {}
    # Write data into the file
    with open(os.path.join(save_path, "checkpoints/",  'output.json'), 'w+') as f:
        json_data[f"{args.exp_name}_{time_label}"] = {'avg_acc': avg_acc, 
                                'ft_avg_acc': ft_avg_acc,
                                'r_c': r_c,
                                'r_r': r_r,
                                'random_seed':seed}
        f.write(json.dumps(json_data))
    logging.info('Finished Training')
    logging.info(f"Command used: {commands}")