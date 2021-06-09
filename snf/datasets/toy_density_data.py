import math
import wandb
import os
import numpy as np
from functools import partial
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from torch.distributions.multivariate_normal import MultivariateNormal

# Implementation modified from: 
# https://github.com/fissoreg/relative-gradient-jacobian/blob/master/experiments/datasets/density.py

def sample_2d_data(dataset, n_samples):
    z = torch.randn(n_samples, 2)

    if dataset == '8gaussians':
        scale = 4
        sq2 = 1/math.sqrt(2)
        centers = [(1,0), (-1,0), (0,1), (0,-1),
                   (sq2,sq2), (-sq2,sq2), (sq2,-sq2), (-sq2,-sq2)]
        centers = torch.tensor([(scale * x, scale * y) for x,y in centers])
        return sq2 * (0.5 * z + centers[torch.randint(len(centers), 
                                                      size=(n_samples,))])

    elif dataset == '1gaussian':
        m = MultivariateNormal(torch.zeros(2), torch.eye(2))
        data = m.rsample(torch.Size([n_samples]))
        return data

    elif dataset == 'sine':
        xs = torch.rand((n_samples, 1)) * 4 - 2
        ys = torch.randn(n_samples, 1) * 0.25

        return torch.cat((xs, torch.sin(3 * xs) + ys), dim=1)

    elif dataset == 'moons':
        from sklearn.datasets import make_moons
        data = make_moons(n_samples=n_samples, shuffle=True, noise=0.05)[0]
        data = torch.tensor(data)
        return data

    elif dataset == 'trimodal':
        centers = torch.tensor([(0, 0), (5, 5), (5, -5)])
        stds = torch.tensor([1., 0.5, 0.5]).unsqueeze(-1)
        seq = torch.randint(len(centers), size=(n_samples,))
        return stds[seq] * z + centers[seq]

    elif dataset == 'trimodal2':
        centers = torch.tensor([(0, 0), (5, 5), (5, -5)])
        stds = torch.tensor([0.5, 0.5, 0.5]).unsqueeze(-1)
        seq = torch.randint(len(centers), size=(n_samples,))
        data = stds[seq] * z + centers[seq]
        return data.unsqueeze(1).unsqueeze(1)

    elif dataset == 'smile':
        scale = 4
        sq2 = 1 / math.sqrt(2)

        # SMILE

        centers = []
        centers.append((scale * 0.5, -scale * 0.8660254037844387))
        centers.append((-scale * 0.5, -scale * 0.8660254037844387))

        centers.append((scale * 0, -scale * 0))

        centers.append((scale * 0, scale * 1))
        centers.append((scale * sq2, scale * sq2))
        centers.append((scale * -sq2, scale * sq2))
        centers.append((scale * 0.5, scale * math.sqrt(3)/2))
        centers.append((scale * 0.25881904510252074, scale * 0.9659258262890683))
        centers.append((-scale * 0.5, scale * math.sqrt(3)/2))
        centers.append((-scale * 0.25881904510252074, scale * 0.9659258262890683))
        centers = torch.tensor(centers)

        weights = torch.tensor([0.5/3, 0.5/3, 0.5/3, 0.5/7, 0.5/7, 
                                0.5/7, 0.5/7, 0.5/7, 0.5/7, 0.5/7])

        stds = torch.tensor([0.5] * len(centers)).unsqueeze(-1)

        from torch.distributions import Categorical
        seq = Categorical(probs=weights).sample((n_samples,))

        return stds[seq] * z + centers[seq]

    elif dataset == '2spirals':
        n = torch.sqrt(torch.rand(n_samples // 2)) * 540 * (2 * math.pi) / 360
        d1x = - torch.cos(n) * n + torch.rand(n_samples // 2) * 0.5
        d1y =   torch.sin(n) * n + torch.rand(n_samples // 2) * 0.5
        x = torch.cat([torch.stack([ d1x,  d1y], dim=1),
                       torch.stack([-d1x, -d1y], dim=1)], dim=0) / 3
        return x + 0.1*z

    elif dataset == 'checkerboard':
        x1 = torch.rand(n_samples) * 4 - 2
        x2_ = torch.rand(n_samples) - torch.randint(0, 2, (n_samples,),
                         dtype=torch.float) * 2
        x2 = x2_ + x1.floor() % 2
        return torch.stack([x1, x2], dim=1) * 2

    elif dataset == 'rings':
        n_samples4 = n_samples3 = n_samples2 = n_samples // 4
        n_samples1 = n_samples - n_samples4 - n_samples3 - n_samples2

        # so as not to have the first point = last point, 
        # set endpoint=False in np; here shifted by one
        linspace4 = torch.linspace(0, 2 * math.pi, n_samples4 + 1)[:-1]
        linspace3 = torch.linspace(0, 2 * math.pi, n_samples3 + 1)[:-1]
        linspace2 = torch.linspace(0, 2 * math.pi, n_samples2 + 1)[:-1]
        linspace1 = torch.linspace(0, 2 * math.pi, n_samples1 + 1)[:-1]

        circ4_x = torch.cos(linspace4)
        circ4_y = torch.sin(linspace4)
        circ3_x = torch.cos(linspace4) * 0.75
        circ3_y = torch.sin(linspace3) * 0.75
        circ2_x = torch.cos(linspace2) * 0.5
        circ2_y = torch.sin(linspace2) * 0.5
        circ1_x = torch.cos(linspace1) * 0.25
        circ1_y = torch.sin(linspace1) * 0.25

        x = torch.stack([torch.cat([circ4_x, circ3_x, circ2_x, circ1_x]),
                         torch.cat([circ4_y, circ3_y, circ2_y, circ1_y])], dim=1) * 3.0

        # random sample
        x = x[torch.randint(0, n_samples, size=(n_samples,))]

        # Add noise
        return x + torch.normal(mean=torch.zeros_like(x), std=0.08*torch.ones_like(x))

    else:
        raise RuntimeError('Invalid `dataset` to sample from.')


def plot(model, potential_or_sampling_fn, name, device, dist_name, dir, n_pts=1000):
    if dist_name == 'smile':
        range_lim = [-6, 6]
    elif dist_name == 'moons':
        range_lim = [-1.5, 2.5]
    elif dist_name == 'sine':
        range_lim = [-2, 2]
    else:
        range_lim = [-8, 8]

    # construct test points
    test_grid = setup_grid(range_lim, n_pts, device)
  
    # plot
    fig, axs = plt.subplots(2, 1, figsize=(4, 8), subplot_kw={'aspect': 'equal'})
    plot_samples(potential_or_sampling_fn, axs[0], range_lim, n_pts)
    plot_fwd_flow_density(model, axs[1], test_grid, n_pts, 100)

    # format
    for ax in plt.gcf().axes: format_ax(ax, range_lim)
    plt.tight_layout()

    # save
    if not os.path.exists(dir):
        print(f'Making dir: {dir}')
        os.makedirs(dir)
    fname = '{}/vis_{}.png'.format(dir, name)
    plt.savefig(fname)
    wandb.log({'Density Plot':  wandb.Image(plt)})
    plt.close()
    return fname

def setup_grid(range_lim, n_pts, device):
    x = torch.linspace(range_lim[0], range_lim[1], n_pts)
    xx, yy = torch.meshgrid((x, x))
    zz = torch.stack((xx.flatten(), yy.flatten()), dim=1)
    return xx, yy, zz.to(device)


def format_ax(ax, range):
    ax.set_xlim(*range)
    ax.set_ylim(*range)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.invert_yaxis()


def plot_samples(samples_fn, ax, range_lim, n_pts):
    samples = samples_fn(n_pts**2).numpy()
    ax.hist2d(samples[:,0], samples[:,1], 
              range=[range_lim, range_lim], bins=n_pts) 
    ax.set_title('Target samples')

def pt_to_tensor(data):
    return data.reshape(data.shape[0], 1, 1, 2)

def plot_fwd_flow_density(model, ax, test_grid, n_pts, batch_size):
    """ plots square grid and flow density """
    xx, yy, zz = test_grid
    data = zz.reshape(-1, 2)
    
    n_batches = data.shape[0] // batch_size
    probs = []

    for b in range(n_batches):
        batch = data[b*batch_size:(b+1)*batch_size]
        log_prob = model.log_prob(batch).cpu().double() 
        prob = torch.exp(log_prob).detach()
        probs.append(prob)

    probs = torch.cat(probs)
    # plot
    ax.pcolormesh(xx, yy, probs.reshape((n_pts,n_pts)))
    ax.set_facecolor(plt.cm.jet(0.))
    ax.set_title('Predicted density')


class ToyDensity(Dataset):
    def __init__(self, name, dataset_size=5000):
        self.label_size = 1
        self.dataset_size = dataset_size
        self.input_size = 2
        self.dataset = sample_2d_data(name, dataset_size)
        self.sample_function = partial(sample_2d_data, name)
        self.name = name

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, i):
        return self.dataset[i], torch.zeros(self.label_size)

    def plot(self, model, name, device, dir, n_pts=1000):
        plot(model, self.sample_function, name, device,
             dist_name=self.name, dir=dir, n_pts=n_pts)
