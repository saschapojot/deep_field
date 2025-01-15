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

#using dropout
t=1

J=16*t
mu=-8.3*t
T=0.1*t
save_interval=25
filter_size=5
# prob_dropout1=0.0
# prob_dropout2=0.01
outCoefDir="./coefs/"
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

class NonlinearLayer(nn.Module):
    #Phi0 to F1
    #T1,T2,...,Tn to g1(T1), g2(T2),...,gn(Tn)
    #S1,S2,...,S_{n-1} to F2=f1(S1),F3=f2(S2),...,Fn=f_{n-1}(S_{n-1})
    #q
    def __init__(self, in_channels, conv1_out_channels, conv2_out_channels, kernel_size, padding=1):
        super().__init__()
        # First convolution
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=conv1_out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )
        self.bn_layer= nn.BatchNorm2d(conv1_out_channels)
        # Tanh activation
        self.tanh = nn.Tanh()
        # Second convolution
        self.conv2 = nn.Conv2d(
            in_channels=conv1_out_channels,
            out_channels=conv2_out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )

        # self.dropout = nn.Dropout2d(p=dropout_prob)  # Dropout applied to feature maps

    def forward(self, x):
        # First convolution
        x = self.conv1(x)  # Shape: (batch_size, conv1_out_channels, N, N)
        # Apply tanh activation
        x=self.bn_layer(x)
        x = self.tanh(x)  # Shape: (batch_size, conv1_out_channels, N, N)

        # Second convolution
        x = self.conv2(x)  # Shape: (batch_size, conv2_out_channels, N, N)


        return x

class dsnn_qt(nn.Module):
    def __init__(self, input_channels,
                 phi0_out_channels, T_out_channels,
                 nonlinear_conv1_out_channels, nonlinear_conv2_out_channels,
                 final_out_channels=1,  # For N x N matrix
                 filter_size=5,stepsAfterInit=1,dropout_prob_inner=0.01,dropout_prob_final=0.1):
        super().__init__()
        paddingNum = (filter_size - 1) // 2
        self.stepsAfterInit = stepsAfterInit
        # Phi0Layer
        self.phi0_layer = Phi0Layer(out_channels=phi0_out_channels, kernel_size=filter_size, padding=paddingNum)
        # T1  Layer
        self.T1_layer = TLayer(out_channels=T_out_channels, kernel_size=filter_size, padding=paddingNum)
        # T layers after init
        self.T_layers_after_init = nn.ModuleList([
            TLayer(out_channels=T_out_channels, kernel_size=filter_size, padding=paddingNum) for _ in
            range(0, stepsAfterInit)
        ])
        # print(f"len(T_layers_after_init)={len(self.T_layers_after_init)}")
        # NonlinearLayer for Phi0and F1
        self.nonlinear_layer_Phi0_2_F1 = NonlinearLayer(
            in_channels=nonlinear_conv2_out_channels,
            conv1_out_channels=nonlinear_conv1_out_channels,
            conv2_out_channels=nonlinear_conv2_out_channels,
            kernel_size=filter_size,  # Default convolution kernel size
            padding=paddingNum
        )

        # f1, ..., f_{n-1}
        self.f_mapping_layers = nn.ModuleList([
            NonlinearLayer(
                in_channels=nonlinear_conv2_out_channels,
                conv1_out_channels=nonlinear_conv1_out_channels,
                conv2_out_channels=nonlinear_conv2_out_channels,
                kernel_size=filter_size,  # Default convolution kernel size
                padding=paddingNum
            )
            for _ in range(0, stepsAfterInit)
        ])
        # print(f"len(f_mapping_layers)={len(self.f_mapping_layers)}")
        # NonlinearLayer for T1 and S1
        self.nonlinear_layer_T1_2_S1 = NonlinearLayer(
            in_channels=T_out_channels,
            conv1_out_channels=nonlinear_conv1_out_channels,
            conv2_out_channels=nonlinear_conv2_out_channels,
            kernel_size=filter_size,
            padding=paddingNum
        )

        # g2,...,gn
        self.g_mapping_layers = nn.ModuleList([
            NonlinearLayer(
                in_channels=nonlinear_conv2_out_channels,
                conv1_out_channels=nonlinear_conv1_out_channels,
                conv2_out_channels=nonlinear_conv2_out_channels,
                kernel_size=filter_size,  # Default convolution kernel size
                padding=paddingNum
            )
            for _ in range(0, stepsAfterInit)
        ])

        # print(f"len(g_mapping_layers)={len(self.g_mapping_layers)}")
        # Final mapping layer to N x N matrix
        self.final_mapping_layer = NonlinearLayer(
            in_channels=nonlinear_conv2_out_channels,
            conv1_out_channels=nonlinear_conv2_out_channels,
            conv2_out_channels=final_out_channels,  # Map to 1 channel for N x N matrix
            kernel_size=3,
            padding=paddingNum
        )

    def initialize_S1(self, x):
        # Step 1: Compute F1 from Phi0Layer and NonlinearLayer
        phi0_output = self.phi0_layer(x)
        #######################
        # #save
        # out_Phi0File = outCoefDir + "/Phi0.pth"
        # torch.save(phi0_output, out_Phi0File)
        # print(f"Phi0 saved to {out_Phi0File}")
        # end save
        ########################


        F1 = self.nonlinear_layer_Phi0_2_F1(phi0_output)

        #######################
        # #save
        # out_F1File = outCoefDir + "/F1.pth"
        # torch.save(F1, out_F1File)
        # print("F1 saved to {}".format(out_F1File))
        # print(f"F1.shape={F1.shape}")
        # end save
        ########################

        # Step 2: Pass input through TLayer and NonlinearLayer

        T_output = self.T1_layer(x)

        #######################
        # #save
        # out_T1File = outCoefDir + "/T1.pth"
        # torch.save(T_output, out_T1File)
        # print("T1 saved to {}".format(out_T1File))
        # print(f"T_output.shape={T_output.shape}")
        # end save
        ########################


        nonlinear_output = self.nonlinear_layer_T1_2_S1(T_output)
        #######################
        # #save
        # out_g1File=outCoefDir+"/g1.pth"
        # torch.save(nonlinear_output, out_g1File)
        # print(f"g1 saved to {out_g1File}")
        # end save
        ########################

        # Step 3: Compute S1 as pointwise multiplication of F1 and nonlinear_output
        S1 = F1 * nonlinear_output

        #######################
        # #save
        # out_S1File = outCoefDir + "/S1.pth"
        # torch.save(S1, out_S1File)
        # print("S1 saved to {}".format(out_S1File))
        # print(f"S1.shape={S1.shape}")
        # end save
        ########################
        return S1

    def forward(self, x, Sn):
        for j in range(0, self.stepsAfterInit):
            # Step 1: Compute F_{n+1} by passing S_n through NonlinearLayer
            Fn_plus_1 = self.f_mapping_layers[j](Sn)

            #######################
            # #save
            # ind = j + 2
            # out_F_file = outCoefDir + f"/F{ind}.pth"
            # torch.save(Fn_plus_1, out_F_file)
            # print(f"F_{ind} saved to {out_F_file}")
            # end save
            ########################

            # Step 2: Pass input through TLayer and NonlinearLayer
            T_output = self.T_layers_after_init[j](x)

            #######################
            # #save
            # out_T_file = outCoefDir + f"/T{ind}.pth"
            # torch.save(T_output, out_T_file)
            # print(f"T{ind} saved to {out_T_file}")
            # end save
            ########################

            nonlinear_output = self.g_mapping_layers[j](T_output)

            #######################
            # #save
            # out_g_file=outCoefDir+f"g{ind}.pth"
            # torch.save(nonlinear_output, out_g_file)
            # print(f"g{ind} saved to {out_g_file}")
            # end save
            ########################

            # Step 3: Compute S_{n+1} as pointwise multiplication of Fn_plus_1 and nonlinear_output
            Sn = Fn_plus_1 * nonlinear_output

            #######################
            # #save
            # out_S_file = outCoefDir + f"/S{ind}.pth"
            # torch.save(Sn, out_S_file)
            # print(f"S{ind} saved to {out_S_file}")
            # end save
            ########################

        # Step 4: Map the final S_{n+1} to N x N matrix
        final_output = self.final_mapping_layer(Sn)
        final_output = final_output.squeeze(1)  # Remove channel dimension (batch_size, 1, N, N) -> (batch_size, N, N)
        # Step 5: Compute the target scalar E by summing all elements in the N x N matrix
        #######################
        # #save
        # out_final_outputFile = outCoefDir + f"/final_output.pth"
        # torch.save(final_output, out_final_outputFile)
        # print(f"final_output saved to {out_final_outputFile}")
        # end save
        ########################
        E = final_output.view(final_output.size(0), -1).sum(dim=1)  # Sum over all elements for each batch
        # print(f"E.shape={E.shape}")
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
