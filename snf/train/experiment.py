import os
import torch
import torchvision
from time import time

try:
    import wandb
except:
    pass

from snf.train.statsrecorder import StatsRecorder


default_config = {
        'name': None,
        'notes': None,
        'wandb': False,
        'wandb_project': 'YOU_PROJECT_NAME',
        'wandb_entity': 'YOUR_ENTITY_NAME',
        'log_timing': False,
        'eval_train': False,
        'max_eval_ex': float('inf'),
        'log_interval': 100,
        'sample_epochs': 10_000,
        'vis_epochs': 10_000,
        'n_samples': 100,
        'sample_dir': 'samples',
        'epochs': 10_000,
        'grad_clip_norm': None,
        'eval_epochs': 1,
        'lr': 1e-3,
        'warmup_epochs': 10,
        'modified_grad': True,
        'add_recon_grad': True,
        'sample_true_inv': False,
        'plot_recon': False,
        'checkpoint_path': None
    }

class Experiment:
    def __init__(self, model, train_loader, val_loader, test_loader,
                 optimizer, scheduler, **kwargs):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler

        try:
            self.data_shape = self.train_loader.dataset.dataset.data.shape[1:]
        except AttributeError:
            if type(train_loader.dataset.dataset) == torchvision.datasets.ImageFolder:
                self.data_shape = train_loader.dataset.dataset[0][0].shape
            else:
                self.data_shape = self.train_loader.dataset.dataset.tensors[0].shape[2:]
        self.to_bpd = lambda x: x / (torch.log(torch.tensor(2.0)) 
                                     * torch.prod(torch.tensor(self.data_shape)))       

        self.config = default_config
        self.config.update(**kwargs)

        self.summary = {}

        if self.config['wandb']:
            wandb.init(name=self.config['name'],
                       notes=self.config['notes'],
                       project=self.config['wandb_project'], 
                       entity=self.config['wandb_entity'], 
                       config=self.config)
            wandb.watch(self.model)

        if self.config['checkpoint_path'] is None and self.config['wandb']:
            self.config['checkpoint_path'] = os.path.join(wandb.run.dir,
                                                          'checkpoint.tar')
        elif self.config['checkpoint_path'] is None:
            checkpoint_path = f"./{str(self.config['name']).replace(' ', '_')}_checkpoint.tar"
            self.log('Warning', f'No checkpoint path specified, defaulting to {checkpoint_path}')
            self.config['checkpoint_path'] = checkpoint_path

        self.update_summary('Epoch', 0)
        self.update_summary("Best Val LogPx", float('-inf'))
        self.update_summary("Test LogPx", float('-inf'))

        if self.config['log_timing']:
            self.batch_time = StatsRecorder()
            self.sample_time = StatsRecorder()

    def run(self):
        for e in range(self.summary['Epoch'] + 1, self.config['epochs'] + 1):
            self.update_summary('Epoch', e)
            avg_loss = self.train_epoch(e)
            self.log('Train Avg Loss', avg_loss)

            if e % self.config['eval_epochs'] == 0:
                if self.config['eval_train']:
                    train_logpx = self.eval_epoch(self.train_loader, e)
                    self.log('Train LogPx', train_logpx)
                    self.log('Train BPD', self.to_bpd(train_logpx))      

                val_logpx = self.eval_epoch(self.val_loader, e, split='Val')
                self.log('Val LogPx', val_logpx)
                self.log('Val BPD', self.to_bpd(val_logpx))
                if val_logpx > self.summary['Best Val LogPx']:
                    self.update_summary('Best Val LogPx', val_logpx)
                    self.update_summary('Best Val BPD', self.to_bpd(val_logpx))
                    test_logpx = self.eval_epoch(self.test_loader, e, split='Test')
                    self.log('Test LogPx', test_logpx)
                    self.log('Test BPD', self.to_bpd(test_logpx))
                    self.update_summary('Test LogPx', test_logpx)
                    self.update_summary('Test BPD', self.to_bpd(test_logpx))

                    # Checkpoint model
                    self.save()

            if e < 5 or e == 10 or e % self.config['sample_epochs'] == 0:
                self.sample(e)

            if e % self.config['vis_epochs'] == 0:
                self.filter_vis()

            self.scheduler.step()

    def log(self, name, val):
        print(f"{name}: {val}")
        if self.config['wandb']: wandb.log({name: val})

    def update_summary(self, name, val):
        print(f"{name}: {val}")
        self.summary[name] = val
        if self.config['wandb']: wandb.run.summary[name] = val

    def get_loss(self, x):
        compute_expensive = not self.config['modified_grad']
        lossval = -self.model.log_prob(x, compute_expensive=compute_expensive)  
        lossval[lossval != lossval] = 0.0 # Replace NaN's with 0      
        lossval = (lossval).sum() / len(x)
        return lossval

    def warmup_lr(self, epoch, num_batches):
        if epoch <= self.config['warmup_epochs']:
            for param_group in self.optimizer.param_groups:
                s = (((num_batches+1) + (epoch-1) * len(self.train_loader)) 
                        / (self.config['warmup_epochs'] * len(self.train_loader)))
                param_group['lr'] = self.config['lr'] * s

    def train_epoch(self, epoch):
        total_loss = 0
        num_batches = 0
        batch_durations = []

        self.model.train()
        for x, _ in self.train_loader:
            self.warmup_lr(epoch, num_batches)
            self.optimizer.zero_grad()
            x = x.float().to('cuda')
            if self.config['log_timing']:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

            lossval = self.get_loss(x)
            lossval.backward()

            if self.config['add_recon_grad']:
                total_recon_loss = self.model.add_recon_grad()
 
            if self.config['grad_clip_norm'] is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                               self.config['grad_clip_norm'])

            self.optimizer.step()

            if self.config['log_timing']:
                end.record()
                torch.cuda.synchronize()
                batch_durations.append(start.elapsed_time(end))

            total_loss += lossval.item()
            num_batches += 1
            if num_batches % self.config['log_interval'] == 0:
                self.log('Train Batch Loss', lossval)
                if self.config['add_recon_grad']:
                    self.log('Train Total Recon Loss', total_recon_loss)

        if self.config['log_timing']:
            # Take all but first 100 and last 100 batch times into account
            self.batch_time.update(batch_durations[100:-100])
            self.update_summary('Batch Time Mean', self.batch_time.mean)
            self.update_summary('Batch Time Std', self.batch_time.std)

        if self.config['plot_recon']:
            self.plot_recon(x, epoch)

        avg_loss = total_loss / num_batches
        return avg_loss

    def eval_epoch(self, dataloader, epoch, split='Val'):
        total_logpx = 0.0
        num_x = 0
        with torch.no_grad():
            self.model.eval()
            for x, _ in dataloader:
                x = x.float().to('cuda')

                total_logpx += self.model.log_prob(x).sum()
                num_x += len(x)
                if num_x >= self.config['max_eval_ex']:
                    break
        avg_logpx = total_logpx / num_x
        return avg_logpx

    def sample(self, e):
        n = self.config['n_samples']
        s_dir = self.config['sample_dir']
        s_path = os.path.join(s_dir, f'{e}.png')
        compute_expensive = not self.config['modified_grad']

        if self.config['log_timing']:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            sample_durations = []
            
            for idx in range(n):
                start.record()
                with torch.no_grad():
                    _, _ = self.model.sample(n_samples=1,
                                compute_expensive=compute_expensive,
                                also_true_inverse=False)
                end.record()
                torch.cuda.synchronize()
                sample_durations.append(start.elapsed_time(end))
            
            self.sample_time.update(sample_durations[n//5:-n//5])
            self.update_summary('Sample Time Mean', self.sample_time.mean)
            self.update_summary('Sample Time Std', self.sample_time.std)

        with torch.no_grad():
            x_sample, x_sample_trueinv = self.model.sample(n_samples=n,
                        compute_expensive=compute_expensive,
                        also_true_inverse=self.config['sample_true_inv']
                        )
            if len(self.data_shape) == 2:
                x_sample = x_sample.view(n, 1, *self.data_shape)
                x_sample_trueinv = x_sample_trueinv.view(n, 1, *self.data_shape)
            else:
                x_sample = x_sample
                x_sample_trueinv = x_sample_trueinv

        os.makedirs(s_dir, exist_ok=True)
        torchvision.utils.save_image(
            x_sample / 256., s_path, nrow=10,
            padding=2, normalize=False)

        if self.config['wandb']:
            wandb.log({'Samples_Approx_Inv':  wandb.Image(s_path)})

        if self.config['sample_true_inv']:
            s_true_inv_path = os.path.join(s_dir, f'{e}_trueinv.png')
            torchvision.utils.save_image(
                        x_sample_trueinv / 256., s_true_inv_path, nrow=10,
                        padding=2, normalize=False)            

            if self.config['wandb']:
                wandb.log({'Samples_True_Inv':  wandb.Image(s_true_inv_path)})

    def filter_vis(self):
        self.model.plot_filters()

    def plot_recon(self, x, e, context=None):
        n = self.config['n_samples']
        s_dir = self.config['sample_dir']
        x_path = os.path.join(s_dir, f'{e}_x.png')
        xhat_path = os.path.join(s_dir, f'{e}_xrecon.png')
        diff_path = os.path.join(s_dir, f'{e}_recon_diff.png')

        compute_expensive = not self.config['modified_grad']

        with torch.no_grad():
            xhat = self.model.reconstruct(x, context, compute_expensive).view(x.shape)

        os.makedirs(s_dir, exist_ok=True)
        torchvision.utils.save_image(
            xhat / 256., xhat_path, nrow=10,
            padding=2, normalize=False)

        torchvision.utils.save_image(
            x / 256., x_path, nrow=10,
            padding=2, normalize=False)

        xdiff = torch.abs(x - xhat)

        torchvision.utils.save_image(
            xdiff / 256., diff_path, nrow=10,
            padding=2, normalize=False)

        if self.config['wandb']:
            wandb.log({'X Original':  wandb.Image(x_path)})
            wandb.log({'X Recon':  wandb.Image(xhat_path)})
            wandb.log({'Recon diff':  wandb.Image(diff_path)})

    def save(self):
        self.log('Note', f'Saving checkpoint to: {self.config["checkpoint_path"]}')
        checkpoint = {
                      'summary': self.summary,
                      'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(),
                      'scheduler_state_dict': self.scheduler.state_dict(),
                      'config': self.config  
                     }

        torch.save(checkpoint, self.config['checkpoint_path'])
        if self.config['wandb']:
            wandb.save(self.config['checkpoint_path'])

    def load(self, path):
        self.log('Note', f'Loading checkpoint from: {path}')
        checkpoint = torch.load(path)

        # Warning, config params overwritten
        self.summary = checkpoint['summary']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        config_diff = set(self.config.items()) ^ set(checkpoint['config'].items())

        if config_diff != set():
            self.log('Warning', f'Differences in loaded config: {config_diff}')
