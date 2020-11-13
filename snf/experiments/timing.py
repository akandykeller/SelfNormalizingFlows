import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import StepLR

from snf.layers.flowsequential import FlowSequential
from snf.layers.selfnorm import SelfNormConv, SelfNormFC
from snf.train.losses import NegativeGaussianLoss
from snf.train.experiment import Experiment


def create_model(data_size, layer='conv'):
    alpha = 1e-6
    layers = []

    c_in = data_size[0]
    h = data_size[1]
    w = data_size[2]

    if layer == 'fc':
        size = c_in * h * w
        layers.append(SelfNormFC(size, size, bias=True,
                                 sym_recon_grad=False, 
                                 only_R_recon=False))
        model = FlowSequential(NegativeGaussianLoss(size=(size,)), *layers)
    
    elif layer == 'conv':
        layers.append(SelfNormConv(c_in, c_in, (3,3), bias=True,
                                 stride=1, padding=1,
                                 sym_recon_grad=False, 
                                 only_R_recon=False))        
        model = FlowSequential(NegativeGaussianLoss(size=(1, 28, 28)), *layers)

    return model


def load_data(**kwargs):
    imsize = (1, 28, 28)

    trainx = torch.randn(60_000, *imsize)
    testx = torch.randn(10_000, *imsize)

    trainy = torch.zeros(60_000)
    testy = torch.zeros(10_000) 

    trainvalset = torch.utils.data.TensorDataset(trainx, trainy)
    testset = torch.utils.data.TensorDataset(testx, testy)

    trainset = torch.utils.data.Subset(trainvalset, range(0, 50_000))
    valset = torch.utils.data.Subset(trainvalset, range(50_000, 60_000))

    train_loader = DataLoader(trainset, **kwargs)
    val_loader = DataLoader(valset, **kwargs)
    test_loader = DataLoader(testset, **kwargs)

    return train_loader, val_loader, test_loader


def main():
    config = {
        'name': 'Timing Experiment SNF FC',
        'eval_epochs': 1,
        'sample_epochs': 1,
        'log_interval': 10000,
        'lr': 1e-4,
        'batch_size': 100,
        'modified_grad': True,
        'add_recon_grad': True,
        'sym_recon_grad': False,
        'only_R_recon': False,
        'actnorm': False,
        'split_prior': False,
        'activation': 'None',
        'log_timing': True,
        'epochs': 100
    }

    train_loader, val_loader, test_loader = load_data(batch_size=config['batch_size'])

    model = create_model(data_size=(1,28,28), layer='fc').to('cuda')

    optimizer = optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.999))
    scheduler = StepLR(optimizer, step_size=1, gamma=1.0)

    experiment = Experiment(model, train_loader, val_loader, test_loader,
                            optimizer, scheduler, **config)

    experiment.run()
