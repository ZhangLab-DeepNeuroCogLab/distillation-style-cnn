#import relevant libraries
from torchvision import datasets, transforms
from torch import nn, optim
import torch
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import time
import copy
import os
from models import *

#define transformations
transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((0.5,), (0.5,))])

#load cifar100 dataset from pytorch
trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
net = resnet32_cifar(num_classes=100).cuda()
net.load_state_dict(torch.load('resnet32_cifar100.pth'))

