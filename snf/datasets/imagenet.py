import torch
import torch.utils.data as data_utils
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from snf.train.datatransforms import ToTensorNoNorm
import tarfile
import os
import math
import numpy as np


def extract_tar(tarpath):
    assert tarpath.endswith('.tar')

    startdir = tarpath[:-4] + '/'

    if os.path.exists(startdir):
        return startdir

    print('Extracting', tarpath)

    with tarfile.open(name=tarpath) as tar:
        t = 0
        done = False
        while not done:
            path = os.path.join(startdir, 'images{}'.format(t))
            os.makedirs(path, exist_ok=True)

            print(path)

            for i in range(50000):
                member = tar.next()

                if member is None:
                    done = True
                    break

                # Skip directories
                while member.isdir():
                    member = tar.next()
                    if member is None:
                        done = True
                        break

                member.name = member.name.split('/')[-1]

                tar.extract(member, path=path)

            t += 1

    return startdir


def load_data(data_aug=False, resolution=32, 
              data_dir='data/imagenet', 
              **kwargs):
    assert resolution == 32 or resolution == 64

    trainpath = f'{data_dir}/train_{resolution}x{resolution}.tar'
    valpath = f'{data_dir}/valid_{resolution}x{resolution}.tar'

    trainpath = extract_tar(trainpath)
    valpath = extract_tar(valpath)

    data_transform = transforms.Compose([
        ToTensorNoNorm()
    ])

    print('Starting loading ImageNet')

    imagenet_data = torchvision.datasets.ImageFolder(
        trainpath,
        transform=data_transform)

    print('Number of data images', len(imagenet_data))

    val_idcs = np.random.choice(len(imagenet_data), size=20000, replace=False)
    train_idcs = np.setdiff1d(np.arange(len(imagenet_data)), val_idcs)

    train_dataset = torch.utils.data.dataset.Subset(
        imagenet_data, train_idcs)
    val_dataset = torch.utils.data.dataset.Subset(
        imagenet_data, val_idcs)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        **kwargs)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        **kwargs)

    test_dataset = torchvision.datasets.ImageFolder(
        valpath,
        transform=data_transform)

    print('Number of val images:', len(test_dataset))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        **kwargs)

    return train_loader, val_loader, test_loader

