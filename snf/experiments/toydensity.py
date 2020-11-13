import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import StepLR

from snf.datasets.toy_density_data import ToyDensity
from snf.layers.flowsequential import FlowSequential
from snf.layers.selfnorm import SelfNormFC
from snf.layers.activations import SmoothLeakyRelu
from snf.train.losses import NegativeGaussianLoss

def create_model(num_layers=100):
    layers = []
    for l in range(num_layers):
        layers.append(SelfNormFC(2, 2, bias=True))
        if l < num_layers - 1:
            layers.append(SmoothLeakyRelu(alpha=0.3))
    return FlowSequential(NegativeGaussianLoss((2,)), *layers)

def load_data(dataset_name, **kwargs):
    dataset = ToyDensity(dataset_name)
    train_loader = torch.utils.data.DataLoader(dataset, **kwargs)
    return dataset, train_loader


def main():
    dataset_name = "moons"
    dataset, dataloader = load_data(dataset_name, batch_size=100)

    model = create_model()
    model.to('cuda')

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=1, gamma=1.0)

    for e in range(6_000):
        total_loss = 0
        total_recon_loss = 0
        num_batches = 0
        for x, _ in dataloader:
            optimizer.zero_grad()
            x = x.float().to('cuda')
            out = -model(x, compute_expensive=False)
            lossval = (out).sum() / len(x)
            lossval.backward()

            # For SNF, add reconstrudction gradient
            total_recon_loss = model.add_recon_grad()

            total_loss += lossval.item()
            total_recon_loss += total_recon_loss.item()
            num_batches += 1
            optimizer.step()
        print(f"Epoch {e}: Total_loss: {total_loss / num_batches}, "
               f"Total Recon: {total_recon_loss / num_batches}")
        scheduler.step()

        if e % 5 == 0:
            print(f"Epoch {e}: Total_loss: {total_loss / num_batches}, "
                    f"Total Recon: {total_recon_loss / num_batches}")
            dataset.plot(model, f'{dataset_name}_epoch{e}', 
                         'cuda', dir='ToyDensitySamples')