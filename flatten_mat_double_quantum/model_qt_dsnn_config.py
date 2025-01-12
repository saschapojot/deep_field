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

#this script defines the neural network, and gives parameters
#correction: more T, f, g

# step_num_after_S1=6
# N=7


t=1

J=16*t
mu=-8.3*t
T=0.1*t


C=25


filter_size=5
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


class NonlinearLayer_with_conv(nn.Module):
    #for f0, g1,g2,...,gn
    def __init__(self, in_channels, conv1_out_channels, kernel_size, padding=2,dropout_prob=0.1,linear_out_features1=128, linear_out_features2=64):
        super().__init__()

        # First convolution
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=conv1_out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )
        # Tanh activation 1
        self.tanh1 = nn.Tanh()

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
        self.tanh2 = nn.Tanh()

        self.linear2 = None  # Placeholder for lazy initialization after tanh2

    def forward(self, x):
        # **First Convolution + Tanh Activation**
        x = self.conv1(x)
        # print(f"x.shape={x.shape}")
        x = self.tanh1(x)
        # **Fractional Pooling**
        x = self.pool(x)
        # print(f"x.shape={x.shape}")
        # **Dropout**
        x = self.dropout(x)

        # **Lazy Initialization of Linear1**
        if self.linear1 is None:
            # Compute in_features based on current tensor shape
            in_features1 = x.size(1) * x.size(2) * x.size(3)  # Channels * Height * Width
            self.linear1 = nn.Linear(in_features1, self.linear_out_features1).to(x.device)
        # **Flatten**
        x = torch.flatten(x, start_dim=1)
        # print(f"x.shape={x.shape}")
        # **Linear1 + Tanh**
        x = self.linear1(x)
        x = self.tanh2(x)
        # **Lazy Initialization of Linear2**

        if self.linear2 is None:
            # Compute in_features dynamically based on linear1's output
            in_features2 = self.linear_out_features1
            self.linear2 = nn.Linear(in_features2, self.linear_out_features2).to(x.device)

        # **Linear2**
        x = self.linear2(x)
        # print(f"x.shape={x.shape}")
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
    def __init__(self,symmetrization_out_channels, symmetrization_kernel_size,
                 stepsAfterInit,
                 nonlinear_conv1_out_channels,nonlinear_conv_kernel_size,nonlinear_conv_dropout_prob,nonlinear_conv_linear_out_features1,nonlinear_conv_linear_out_features2,
                 no_conv_lin_layer1_in_dim,no_conv_lin_layer1_out_dim,no_conv_lin_layer2_out_dim):
        super().__init__()

        symmetrization_padding_num=(symmetrization_kernel_size-1)//2
        self.stepsAfterInit = stepsAfterInit
        #Phi0 layer
        self.Ph0_layer=Phi0Layer(out_channels=symmetrization_out_channels,kernel_size=symmetrization_kernel_size,padding=symmetrization_padding_num)

        # T1  Layer
        self.T1_layer=TLayer(out_channels=symmetrization_out_channels,kernel_size=symmetrization_kernel_size,padding=symmetrization_padding_num)
        # T layers after init, T2, T3, ..., T_{n-1}
        self.T_layers_after_init = nn.ModuleList([
            TLayer(out_channels=symmetrization_out_channels, kernel_size=symmetrization_kernel_size,
                   padding=symmetrization_padding_num) for _ in range(0,stepsAfterInit)
            ])

        print(f"len(T_layers_after_init)={len(self.T_layers_after_init)}")

        #NonlinearLayer_with_conv(nn.Module): for Phi0 and F1, i.e., f0
        nonlinear_conv_paddingNum=(nonlinear_conv_kernel_size-1)//2
        self.nonlinear_layer_with_conv_Phi0_2_F1=NonlinearLayer_with_conv(in_channels=symmetrization_out_channels,conv1_out_channels=nonlinear_conv1_out_channels,kernel_size=nonlinear_conv_kernel_size,padding=nonlinear_conv_paddingNum,dropout_prob=nonlinear_conv_dropout_prob,linear_out_features1=nonlinear_conv_linear_out_features1,linear_out_features2=nonlinear_conv_linear_out_features2)

        # NonlinearLayer for T1 and S1, i.e., g1
        self.nonlinear_layer_with_conv_T1_2_S1=NonlinearLayer_with_conv(in_channels=symmetrization_out_channels,conv1_out_channels=nonlinear_conv1_out_channels,kernel_size=nonlinear_conv_kernel_size,padding=nonlinear_conv_paddingNum,dropout_prob=nonlinear_conv_dropout_prob,linear_out_features1=nonlinear_conv_linear_out_features1,linear_out_features2=nonlinear_conv_linear_out_features2)

        # f1, ..., f_{n-1}
        self.f_mapping_layers = nn.ModuleList([
            Nonlinear_layer_without_conv(lin_layer1_in_dim=no_conv_lin_layer1_in_dim,lin_layer1_out_dim=no_conv_lin_layer1_out_dim,lin_layer2_out_dim=no_conv_lin_layer2_out_dim)
            for _ in range(0, stepsAfterInit)
            ])

        print(f"len(f_mapping_layers)={len(self.f_mapping_layers)}")
        # g2,...,gn
        self.g_mapping_layers = nn.ModuleList([
            NonlinearLayer_with_conv(in_channels=symmetrization_out_channels,conv1_out_channels=nonlinear_conv1_out_channels,kernel_size=nonlinear_conv_kernel_size,padding=nonlinear_conv_paddingNum,dropout_prob=nonlinear_conv_dropout_prob,linear_out_features1=nonlinear_conv_linear_out_features1,linear_out_features2=nonlinear_conv_linear_out_features2)
            for _ in range(0, stepsAfterInit)
            ])
        print(f"len(g_mapping_layers)={len(self.g_mapping_layers)}")


    def initialize_S1(self, x):
        """
                Initializes S1 using the input x.

                Args:
                    x (torch.Tensor): Input tensor of shape (batch_size, input_channels, N, N).

                Returns:
                    torch.Tensor: The tensor S1.
                """
        # Step 1: Compute F1
        Phi0_output=self.Ph0_layer(x)
        F1=self.nonlinear_layer_with_conv_Phi0_2_F1(Phi0_output)
        # print(f"F1.shape={F1.shape}")

        # Step 2: Pass input through TLayer and NonlinearLayer
        T1_output=self.T1_layer(x)
        # print(f"T1_output.shape={T1_output.shape}")
        nonlin_conv_output=self.nonlinear_layer_with_conv_T1_2_S1(T1_output)
        # Step 3: Compute S1 as pointwise multiplication of F1 and nonlinear_output
        S1 = F1 *nonlin_conv_output

        # print(f"S1.shape={S1.shape}")

        return S1

    def forward(self, x, Sn):
        for j in range(0, self.stepsAfterInit):
            # Step 1: Compute F_{n+1} by passing S_n through Nonlinear_layer_without_conv
            Fn_plus_1 = self.f_mapping_layers[j](Sn)
            print(f"Fn_plus_1.shape={Fn_plus_1.shape}")
            # Step 2: Pass input through TLayer and NonlinearLayer_with_conv
            T_output = self.T_layers_after_init[j](x)
            nonlinear_output = self.g_mapping_layers[j](T_output)
            # Step 3: Compute S_{n+1} as pointwise multiplication of Fn_plus_1 and nonlinear_output
            Sn = Fn_plus_1 * nonlinear_output
            print(f"Sn.shape={Sn.shape}")
        E = Sn.sum(dim=1, keepdim=True)  # Sum over all elements for each sample
        return E








class CustomDataset(Dataset):
    def __init__(self, X, Y):
        """
        Custom dataset for supervised learning with `dsnn_qt`.

        Args:
            X (torch.Tensor): Input tensor of shape (num_samples, 3, N, N).
            Y (torch.Tensor): Target tensor of shape (num_samples,).
        """
        self.X = X
        self.Y = Y

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.Y)

    def __getitem__(self, idx):
        """
        Retrieves the input and target at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (input, target) where input is of shape (3, N, N) and target is a scalar.
        """
        return self.X[idx], self.Y[idx]
