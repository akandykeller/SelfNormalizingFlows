import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR

from snf.layers import Dequantization, Normalization
from snf.layers.distributions.uniform import UniformDistribution
from snf.layers.splitprior import SplitPrior
from snf.layers.flowsequential import FlowSequential
from snf.layers.selfnorm import SelfNormConv
from snf.layers.actnorm import ActNorm
from snf.layers.squeeze import Squeeze
from snf.layers.transforms import LogitTransform
from snf.layers.coupling import Coupling
from snf.train.losses import NegativeGaussianLoss
from snf.train.experiment import Experiment
from snf.datasets.mnist import load_data


def create_model(num_blocks=2, block_size=16, sym_recon_grad=False, 
                 actnorm=False, split_prior=False, recon_loss_weight=100.0):
    alpha = 1e-6
    layers = [
        Dequantization(UniformDistribution(size=(1, 28, 28))),
        Normalization(translation=0, scale=256),
        Normalization(translation=-alpha, scale=1 / (1 - 2 * alpha)),
        LogitTransform(),
    ]

    current_size = (1, 28, 28)

    for l in range(num_blocks):
        layers.append(Squeeze())
        current_size = (current_size[0]*4, current_size[1]//2, current_size[2]//2)

        for k in range(block_size):
            if actnorm:
                layers.append(ActNorm(current_size[0]))
            
            layers.append(SelfNormConv(current_size[0], current_size[0], (1, 1), 
                                       bias=True, stride=1, padding=0,
                                       sym_recon_grad=sym_recon_grad, 
                                       recon_loss_weight=recon_loss_weight))
            layers.append(Coupling(current_size))

        if split_prior and l < num_blocks - 1:
            layers.append(SplitPrior(current_size, NegativeGaussianLoss))
            current_size = (current_size[0] // 2, current_size[1], current_size[2])

    return FlowSequential(NegativeGaussianLoss(size=current_size), 
                         *layers)


def main():
    config = {
        'name': '2L-16K Glow SNF recon 100x MNIST',
        'eval_epochs': 1,
        'sample_epochs': 1,
        'log_interval': 100,
        'lr': 1e-3,
        'num_blocks': 2,
        'block_size': 16,
        'batch_size': 100,
        'modified_grad': True,
        'add_recon_grad': True,
        'sym_recon_grad': False,
        'actnorm': True,
        'split_prior': True,
        'activation': 'None',
        'recon_loss_weight': 100.0,
        'sample_true_inv': True,
        'plot_recon': True,
        'vis_epochs': 1
    }

    train_loader, val_loader, test_loader = load_data(data_aug=True, batch_size=config['batch_size'])

    model = create_model(num_blocks=config['num_blocks'],
                         block_size=config['block_size'], 
                         sym_recon_grad=config['sym_recon_grad'],
                         actnorm=config['actnorm'],
                         split_prior=config['split_prior'],
                         recon_loss_weight=config['recon_loss_weight']).to('cuda')

    optimizer = optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.999))
    scheduler = StepLR(optimizer, step_size=1, gamma=1.0)

    experiment = Experiment(model, train_loader, val_loader, test_loader,
                            optimizer, scheduler, **config)

    experiment.run()