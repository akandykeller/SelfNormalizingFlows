import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
from functools import reduce

from snf.layers import Dequantization, Normalization
from snf.layers.distributions.uniform import UniformDistribution
from snf.layers.flowsequential import FlowSequential
from snf.layers.selfnorm import SelfNormFC
from snf.layers.activations import SmoothLeakyRelu, SplineActivation, LearnableLeakyRelu
from snf.layers.squeeze import Squeeze
from snf.layers.transforms import LogitTransform
from snf.train.losses import NegativeGaussianLoss
from snf.train.experiment import Experiment
from snf.datasets.mnist import load_data

activations = {
    'SLR': lambda size: SmoothLeakyRelu(alpha=0.3),
    'LLR': lambda size: LearnableLeakyRelu(),
    'Spline': lambda size: SplineActivation(size, tail_bound=10, individual_weights=True),
}

def create_model(num_layers=2, sym_recon_grad=False, 
                 only_R_recon=False,
                 activation='Spline',
                 recon_loss_weight=1.0,
                 data_size=(1, 28, 28)):
    alpha = 1e-6
    layers = [
        Dequantization(UniformDistribution(size=data_size)),
        Normalization(translation=0, scale=256),
        Normalization(translation=-alpha, scale=1 / (1 - 2 * alpha)),
        LogitTransform(),
    ]
    act = activations[activation]

    size = reduce(lambda x,y: x*y, data_size)

    for l in range(num_layers):
        layers.append(SelfNormFC(size, size, bias=True,
                                 sym_recon_grad=sym_recon_grad, 
                                 only_R_recon=only_R_recon,
                                 recon_loss_weight=recon_loss_weight))
        if (l+1) < num_layers:
            layers.append(act((size,)))

    return FlowSequential(NegativeGaussianLoss(size=(size,)), *layers)


def main():
    config = {
        'name': '2L FC SNF MNIST',
        'eval_epochs': 1,
        'sample_epochs': 1,
        'log_interval': 100,
        'lr': 1e-4,
        'num_layers': 2,
        'batch_size': 100,
        'modified_grad': True,
        'add_recon_grad': True,
        'sym_recon_grad': False,
        'only_R_recon': False,
        'activation': 'SLR',
        'recon_loss_weight': 1.0,
        'sample_true_inv': False,
        'plot_recon': False,
        'log_timing': True
    }

    train_loader, val_loader, test_loader = load_data(batch_size=config['batch_size'])

    model = create_model(num_layers=config['num_layers'], 
                         sym_recon_grad=config['sym_recon_grad'],
                         only_R_recon=config['only_R_recon'],
                         activation=config['activation'],
                         recon_loss_weight=config['recon_loss_weight']).to('cuda')

    optimizer = optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.999))
    scheduler = StepLR(optimizer, step_size=1, gamma=1.0)

    experiment = Experiment(model, train_loader, val_loader, test_loader,
                            optimizer, scheduler, **config)

    experiment.run()
