import torch
import torch.nn as nn

import pickle
from decimal import Decimal, getcontext
import numpy as np
from datetime import datetime
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


def format_using_decimal(value, precision=10):
    # Set the precision higher to ensure correct conversion
    getcontext().prec = precision + 2
    # Convert the float to a Decimal with exact precision
    decimal_value = Decimal(str(value))
    # Normalize to remove trailing zeros
    formatted_value = decimal_value.quantize(Decimal(1)) if decimal_value == decimal_value.to_integral() else decimal_value.normalize()
    return str(formatted_value)



class DSNN(nn.Module):
    def __init__(self, num_spins, num_layers, num_neurons):
        """
                Deep Self-Learning Neural Network for Random Energy Model.

                Args:
                    num_spins (int): Number of spins in the system (input size).
                    num_layers (int): Number of self-learning (SL) layers.
                    num_neurons (int): Number of neurons in each SL layer.
        """
        super(DSNN, self).__init__()

        self.num_layers = num_layers

        # Effective field layers (F_i)
        self.effective_field_layers = nn.ModuleList(
            [nn.Sequential(
            nn.Linear(num_spins if i==1 else num_neurons , num_neurons),  # Optional intermediate transformation
            nn.Tanh(),
            nn.Linear(num_neurons, num_neurons)
        ) for i in range(1,num_layers+1)]
        )

        # Quasi-particle layers (S_i)
        self.quasi_particle_layers = nn.ModuleList(
            [nn.Linear(num_spins,num_neurons) for i in range(1,num_layers+1)]
        )

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(num_neurons, num_neurons),  # Optional intermediate transformation
            nn.Tanh(),
            nn.Linear(num_neurons, 1)  # Final mapping to scalar
        )

    def forward(self, S0):
        """

        Forward pass through DSNN.

        Args:
            S0 (torch.Tensor): Input spin configurations, shape (batch_size, num_spins).

        Returns:
            torch.Tensor: Predicted energy, shape (batch_size, 1).
        """

        # Initialize S as the input spin configuration
        S = S0

        for i in range(1,self.num_layers+1):
            # Compute effective field layer Fi
            Fi = self.effective_field_layers[i-1](S)

            # Compute quasi-particle layer Si
            Si = torch.tanh(self.quasi_particle_layers[i-1](S0)) * Fi

            # Update S for the next layer
            S = Si

        # Output layer to compute energy
        E = self.output_layer(S).sum(dim=1, keepdim=True)  # Ensure scalar output per sample

        return E


# Define a custom dataset
class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
# System Parameters
L = 15  # Number of spins
r = 2   # Number of spins in each interaction term
# Reduce learning rate by a factor of gamma every step_size epochs
decrease_over=70
num_layers = 8  # Number of DSNN layers