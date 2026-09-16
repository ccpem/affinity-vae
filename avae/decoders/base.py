import abc
import enum

import torch
import torch.nn as nn


# Abstract Decoder
class AbstractDecoder(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, x: torch.Tensor, x_pose: torch.Tensor) -> torch.Tensor:
        pass
