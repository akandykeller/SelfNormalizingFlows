import torch
import torch.nn as nn
from .flowlayer import ModifiedGradFlowLayer
from .flowlayer import PreprocessingFlowLayer
from .activations import FlowActivationLayer
from .selfnorm import SelfNormConv

class FlowSequential(nn.Module):
    def __init__(self, base_distribution, *modules):
        super().__init__()
        self.base_distribution = base_distribution

        for i, module in enumerate(modules):
            self.add_module(str(i), module)
        self.sequence_modules = modules

    def __iter__(self):
        yield from self.sequence_modules

    def forward(self, input, context=None, compute_expensive=False):
        return self.log_prob(input, context, compute_expensive)

    def log_prob(self, input, context=None, compute_expensive=True):
        logdet = 0
        for idx, module in enumerate(self):
            if isinstance(module, ModifiedGradFlowLayer):
                output, layer_logdet = module(
                    input, context, compute_expensive=compute_expensive)
            else:
                output, layer_logdet = module(input, context)

            logdet += layer_logdet

            # Important to keep track of module input/output for logdet
            # calculation.
            input = output

        logprob = self.base_distribution.log_prob(input)
        
        return logprob + logdet

    def cheap_unnormed_log_prob(self, input, context=None):
        return self.log_prob(input, context=context, compute_expensive=False)

    def activation_modules(self):
        for module in self.sequence_modules:
            if isinstance(module, FlowActivationLayer):
                yield module

    def selfnorm_modules(self):
        for module in self.sequence_modules:
            if isinstance(module, SelfNormConv):
                yield module

    def preprocessing_modules(self):
        for module in self.sequence_modules:
            if isinstance(module, PreprocessingFlowLayer):
                yield module

    def non_preprocessing_modules(self):
        for module in self.sequence_modules:
            if not isinstance(module, PreprocessingFlowLayer):
                yield module

    def non_preprocessing_logdet(self, input, context=None, *, compute_expensive=False):
        logdet = 0
        for module in self.non_preprocessing_modules():
            output, layer_logdet = module(input, context, compute_expensive)
            input = output
            logdet = logdet + layer_logdet
        logprob = self.base_distribution.log_prob(input)
        return logprob + logdet

    def add_recon_grad(self):
        total_recon_loss = 0.0

        for idx, conv in enumerate(self.selfnorm_modules()):
            layer_recon_loss = conv.add_recon_grad()
            total_recon_loss += layer_recon_loss
        return total_recon_loss

    def sample(self, n_samples, context=None, compute_expensive=False, 
                also_true_inverse=False):
        z, logprob = self.base_distribution.sample(n_samples, context)
        
        # Regular sample
        input = z
        for module in reversed(self.sequence_modules):
            if isinstance(module, ModifiedGradFlowLayer):
                output = module.reverse(input, context, compute_expensive)
            else:
                output = module.reverse(input, context)
            input = output

        # True inverse if applicable
        if not compute_expensive and also_true_inverse:
            input_true = z
            for module in reversed(self.sequence_modules):
                if isinstance(module, ModifiedGradFlowLayer):
                    output = module.reverse(input_true, context, 
                                            compute_expensive=True)
                else:
                    output = module.reverse(input_true, context)
                input_true = output
        else:
            input_true = input

        return input, input_true

    def reconstruct(self, x, context=None, compute_expensive=False):
        input = x

        # Forward
        for module in self.sequence_modules:
            if isinstance(module, ModifiedGradFlowLayer):
                output, layer_logdet = module(
                    input, context, compute_expensive=compute_expensive)
            else:
                output, layer_logdet = module(input, context)

            input = output
        
        # Reverse
        for module in reversed(self.sequence_modules):
            if isinstance(module, ModifiedGradFlowLayer):
                output = module.reverse(input, context, compute_expensive)
            else:
                output = module.reverse(input, context)
            input = output
        
        return input