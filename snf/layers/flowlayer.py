from abc import ABCMeta, abstractmethod
from functools import wraps

import torch.nn as nn


class FlowLayer(nn.Module, metaclass=ABCMeta):

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        return instance

    @abstractmethod
    def forward(self, input, context=None):
        pass

    @abstractmethod
    def reverse(self, input, context=None):
        pass

    @abstractmethod
    def logdet(self, input, context=None):
        pass


class ModifiedGradFlowLayer(FlowLayer):

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        return instance

    @abstractmethod
    def forward(self, input, context=None, compute_expensive=False):
        pass

    @abstractmethod
    def reverse(self, input, context=None, compute_expensive=False):
        pass

    @abstractmethod
    def logdet(self, input, context=None, compute_expensive=False):
        pass


class PreprocessingFlowLayer(FlowLayer):
    pass


def mark_expensive(func):
    func._expensive_computation = True
    return func
