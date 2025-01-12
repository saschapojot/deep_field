import torch
import torch.nn as nn
from decimal import Decimal, getcontext
import pickle
from decimal import Decimal, getcontext
import numpy as np
from datetime import datetime
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys

t=1

J=16*t
mu=-8.3*t
T=0.1*t

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def format_using_decimal(value, precision=10):
    # Set the precision higher to ensure correct conversion
    getcontext().prec = precision + 2
    # Convert the float to a Decimal with exact precision
    decimal_value = Decimal(str(value))
    # Normalize to remove trailing zeros
    formatted_value = decimal_value.quantize(Decimal(1)) if decimal_value == decimal_value.to_integral() else decimal_value.normalize()
    return str(formatted_value)


class Phi0Layer(nn.Module):
    def __init__(self, out_channels, kernel_size, padding=2):
        """
        A modified Phi0Layer where the convolutional matrices (weights) are shared
        across the three input channels (Sigma_x, Sigma_y, Sigma_z).

        Args:
            out_channels (int): Number of output channels per input channel.
            kernel_size (int): Size of the convolution kernel.
            padding (int): Padding for the convolution operation. Default is 2.
        """
        super().__init__()

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        # Shared convolutional layers for W0 and W1
        self.shared_conv_W0 = nn.Conv2d(
            in_channels=1,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=False
        )

        self.shared_conv_W1 = nn.Conv2d(
            in_channels=1,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=False
        )

    def forward(self, x):
        """
        Forward pass for the Phi0Layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, N, N).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, N, N).
        """
        # Split input into three individual channels
        x_channels = torch.chunk(x, chunks=3, dim=1)  # [(batch_size, 1, N, N), ...]

        # Apply shared W0 and W1 convolutions to each channel
        conv_W0_outputs = [self.shared_conv_W0(channel) for channel in x_channels]
        conv_W1_outputs = [self.shared_conv_W1(channel) for channel in x_channels]

        # Perform element-wise multiplication for each channel
        multiplied_outputs = [
            conv_W0 * conv_W1 for conv_W0, conv_W1 in zip(conv_W0_outputs, conv_W1_outputs)
        ]  # [(batch_size, out_channels, N, N), ...]

        # Sum over the 3 input channels
        Phi0 = sum(multiplied_outputs)  # Shape: (batch_size, out_channels, N, N)

        return Phi0



class TLayer(nn.Module):
    def __init__(self, out_channels, kernel_size, padding=2):
        """
        A layer with the same functionality as Phi0Layer:
        - Applies shared convolutional weights across three input channels.
        - Performs element-wise multiplication between convolution results.
        - Sums the results over the three input channels.

        Args:
            out_channels (int): Number of output channels per input channel.
            kernel_size (int): Size of the convolution kernel.
            padding (int): Padding for the convolution operation. Default is 2.
        """
        super().__init__()

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        # Shared convolutional layers for W0 and W1
        self.shared_conv_W0 = nn.Conv2d(
            in_channels=1,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=False
        )

        self.shared_conv_W1 = nn.Conv2d(
            in_channels=1,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=False
        )

    def forward(self, x):
        """
        Forward pass for the TLayer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, N, N).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, N, N).
        """
        # Split input into three individual channels
        x_channels = torch.chunk(x, chunks=3, dim=1)  # [(batch_size, 1, N, N), ...]

        # Apply shared W0 and W1 convolutions to each channel
        conv_W0_outputs = [self.shared_conv_W0(channel) for channel in x_channels]
        conv_W1_outputs = [self.shared_conv_W1(channel) for channel in x_channels]

        # Perform element-wise multiplication for each channel
        multiplied_outputs = [
            conv_W0 * conv_W1 for conv_W0, conv_W1 in zip(conv_W0_outputs, conv_W1_outputs)
        ]  # [(batch_size, out_channels, N, N), ...]

        # Sum over the 3 input channels
        T_result = sum(multiplied_outputs)  # Shape: (batch_size, out_channels, N, N)

        return T_result


class nonlinear_symmetry_2_flatten(nn.Module):
    # for f0, g1,g2,...,gn
    def __init__(self, dropout_prob=0.1,
                 linear_out_features1=128, linear_out_features2=64):
        super().__init__()
        # **Fractional Pooling Layer**

        self.pool = nn.FractionalMaxPool2d(
            kernel_size=2,  # Size of the pooling region
            output_ratio=(2 / 3, 2 / 3)  # Fractional reduction ratio
        )
        # **Dropout Layer**
        self.dropout = nn.Dropout2d(p=dropout_prob)  # Dropout applied to feature maps
        # **Flatten Layer**
        self.flatten = nn.Flatten()  # Flattens feature maps to a 1D vector
        self.linear_out_features1 = linear_out_features1  # Output features for linear1
        self.linear_out_features2 = linear_out_features2  # Output features for linear2
        self.linear1 = None  # Placeholder for lazy initialization

        # **Second Tanh Activation**
        self.tanh = nn.Tanh()

        self.linear2 = None  # Placeholder for lazy initialization after tanh2

    def forward(self, x):
        x = self.pool(x)
        x = self.dropout(x)
        # **Lazy Initialization of Linear1**
        if self.linear1 is None:
            # Compute in_features based on current tensor shape
            in_features1 = x.size(1) * x.size(2) * x.size(3)  # Channels * Height * Width
            self.linear1 = nn.Linear(in_features1, self.linear_out_features1).to(x.device)
        # **Flatten**
        x = torch.flatten(x, start_dim=1)
        x = self.linear1(x)
        x = self.tanh(x)
        # **Lazy Initialization of Linear2**
        if self.linear2 is None:
            # Compute in_features dynamically based on linear1's output
            in_features2 = self.linear_out_features1
            self.linear2 = nn.Linear(in_features2, self.linear_out_features2).to(x.device)

        x = self.linear2(x)
        return x


class Nonlinear_layer_without_conv(nn.Module):
    #for f1,f2,...,f_{n-1}
    def __init__(self,lin_layer1_in_dim,lin_layer1_out_dim,lin_layer2_out_dim):
        super().__init__()

        self.lin_layer1 = nn.Linear(lin_layer1_in_dim, lin_layer1_out_dim)
        self.tanh=nn.Tanh()
        self.lin_layer2 = nn.Linear(lin_layer1_out_dim, lin_layer2_out_dim)
    def forward(self, x):
        x=self.lin_layer1(x)
        x=self.tanh(x)
        x=self.lin_layer2(x)
        return x

class DSNN_qt(nn.Module):
    def __init__(self,symmetrization_out_channels,stepsAfterInit,)
        super().__init__()
