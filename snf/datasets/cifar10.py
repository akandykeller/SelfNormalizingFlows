import torch
import torch.utils.data as data_utils
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from snf.train.datatransforms import ToTensorNoNorm
import math
import numpy as np

def load_data(data_aug=True, **kwargs):
    assert data_aug == True
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Pad(int(math.ceil(32 * 0.04)), padding_mode='edge'),
        transforms.RandomAffine(degrees=0, translate=(0.04, 0.04)),
        transforms.CenterCrop(32),
        transforms.RandomHorizontalFlip(),
        ToTensorNoNorm()
    ])

    test_transform = transforms.Compose([
        ToTensorNoNorm()
    ])

    data_train = torchvision.datasets.CIFAR10('./data', train=True, transform=train_transform,
                                              target_transform=None, download=True)

    train = torch.utils.data.Subset(data_train, torch.arange(0, 40000))

    data_val = torchvision.datasets.CIFAR10('./data', train=True,
                                              transform=test_transform,
                                              target_transform=None,
                                              download=False)

    val = torch.utils.data.Subset(data_val, torch.arange(40000, 50000))

    test = torchvision.datasets.CIFAR10('./data', train=False,
                                          transform=test_transform,
                                          target_transform=None,
                                          download=True)

    train_loader = data_utils.DataLoader(train, 
                                         shuffle=True, **kwargs)

    val_loader = data_utils.DataLoader(val, 
                                       shuffle=False, **kwargs)

    test_loader = data_utils.DataLoader(test,
                                        shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader
