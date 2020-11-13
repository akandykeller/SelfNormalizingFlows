import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from snf.train.datatransforms import ToTensorNoNorm

def load_data(**kwargs):
    transform = transforms.Compose([
        ToTensorNoNorm()
    ])
    trainvalset = datasets.MNIST('../data',
                                 train=True,
                                 download=True,
                                 transform=transform)
    testset = datasets.MNIST('../data', train=False, transform=transform)

    trainset = torch.utils.data.Subset(trainvalset, range(0, 50_000))
    valset = torch.utils.data.Subset(trainvalset, range(50_000, 60_000))

    train_loader = DataLoader(trainset, **kwargs)
    val_loader = DataLoader(valset, **kwargs)
    test_loader = DataLoader(testset, **kwargs)

    return train_loader, val_loader, test_loader