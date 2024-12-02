import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from decimal import Decimal, getcontext
import numpy as np
from datetime import datetime

from torch.utils.data import Dataset, DataLoader


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
            [nn.Linear(num_spins,num_spins) for i in range(num_layers)]
        )

        # Quasi-particle layers (S_i)
        self.quasi_particle_layers = nn.ModuleList(
            [nn.Linear(num_spins,num_spins) for i in range(num_layers)]
        )

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(num_spins, num_spins),  # Optional intermediate transformation
            nn.Tanh(),
            nn.Linear(num_spins, 1)  # Final mapping to scalar
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

        for i in range(self.num_layers):
            # Compute effective field layer Fi
            Fi = torch.tanh(self.effective_field_layers[i](S))

            # Compute quasi-particle layer Si
            Si = torch.tanh(self.quasi_particle_layers[i](S0)) * Fi

            # Update S for the next layer
            S = Si

        # Output layer to compute energy
        E = self.output_layer(S).sum(dim=1, keepdim=True)  # Ensure scalar output per sample

        return E


T=1.5

TStr=format_using_decimal(T)
inDir=f"./data_rand_energy_spin_T_{TStr}/"
fileNameTrain=inDir+"/rand_energy_spin.train.pkl"

# Load data from pickle
with open(fileNameTrain, 'rb') as f:
    X_train, Y_train = pickle.load(f)


num_sample,num_spins=X_train.shape

# Hyperparameters
batch_size = 64
learning_rate = 0.001
weight_decay = 0.01  # L2 regularization strength
num_epochs = int(num_sample/batch_size*2)
# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device="+str(device))
# Load data
X_train = torch.tensor(X_train, dtype=torch.float64).to(device)  # Move to device
Y_train = torch.tensor(Y_train, dtype=torch.float64).view(-1, 1).to(device)
# Define a custom dataset
class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# Create Dataset and DataLoader
dataset = CustomDataset(X_train, Y_train)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

num_layers = 4  # Number of DSNN layers
num_neurons = num_spins  # Number of neurons per layer

model = DSNN(num_spins=num_spins, num_layers=num_layers, num_neurons=num_neurons).double().to(device)
# Define loss function and optimizer with L2 regularization
# Optimizer and loss
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Training loop
tTrainStart = datetime.now()
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for X_batch, Y_batch in dataloader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)  # Move batch to device

        # Forward pass
        predictions = model(X_batch)

        # Compute loss
        loss = criterion(predictions, Y_batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate batch loss
        # print(f"loss.item()={loss.item()}")
        # print(f"X_batch.size(0)={X_batch.size(0)}")
        epoch_loss += loss.item() * X_batch.size(0)
        # print(f"loss.item() * X_batch.size(0)={loss.item() * X_batch.size(0)}")

    # Average loss over total samples
    # print(f"len(dataset)={len(dataset)}")
    average_loss = epoch_loss / len(dataset)

    # Print and log epoch summary
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}")
    with open(inDir + "/training_log.txt", "a") as log_file:
        log_file.write(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}\n")

# Save the model
torch.save(model.state_dict(), inDir + "/model.pth")
tTrainEnd = datetime.now()

print("Training time:", tTrainEnd - tTrainStart)