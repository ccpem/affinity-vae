import abc
import enum

import torch
import torch.nn as nn


# Abstract Encoder
class AbstractEncoder(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, x):
        pass
