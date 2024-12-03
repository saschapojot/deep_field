import torch
import torch.nn as nn

import pickle
from decimal import Decimal, getcontext
import numpy as np
from datetime import datetime
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from model_dsnn_config import format_using_decimal,DSNN,CustomDataset,L,r,decrease_over,num_layers


# def format_using_decimal(value, precision=10):
#     # Set the precision higher to ensure correct conversion
#     getcontext().prec = precision + 2
#     # Convert the float to a Decimal with exact precision
#     decimal_value = Decimal(str(value))
#     # Normalize to remove trailing zeros
#     formatted_value = decimal_value.quantize(Decimal(1)) if decimal_value == decimal_value.to_integral() else decimal_value.normalize()
#     return str(formatted_value)



# class DSNN(nn.Module):
#     def __init__(self, num_spins, num_layers, num_neurons):
#         """
#                 Deep Self-Learning Neural Network for Random Energy Model.
#
#                 Args:
#                     num_spins (int): Number of spins in the system (input size).
#                     num_layers (int): Number of self-learning (SL) layers.
#                     num_neurons (int): Number of neurons in each SL layer.
#         """
#         super(DSNN, self).__init__()
#
#         self.num_layers = num_layers
#
#         # Effective field layers (F_i)
#         self.effective_field_layers = nn.ModuleList(
#             [nn.Sequential(
#             nn.Linear(num_spins if i==1 else num_neurons , num_neurons),  # Optional intermediate transformation
#             nn.Tanh(),
#             nn.Linear(num_neurons, num_neurons)
#         ) for i in range(1,num_layers+1)]
#         )
#
#         # Quasi-particle layers (S_i)
#         self.quasi_particle_layers = nn.ModuleList(
#             [nn.Linear(num_spins,num_neurons) for i in range(1,num_layers+1)]
#         )
#
#         # Output layer
#         self.output_layer = nn.Sequential(
#             nn.Linear(num_neurons, num_neurons),  # Optional intermediate transformation
#             nn.Tanh(),
#             nn.Linear(num_neurons, 1)  # Final mapping to scalar
#         )
#
#     def forward(self, S0):
#         """
#
#         Forward pass through DSNN.
#
#         Args:
#             S0 (torch.Tensor): Input spin configurations, shape (batch_size, num_spins).
#
#         Returns:
#             torch.Tensor: Predicted energy, shape (batch_size, 1).
#         """
#
#         # Initialize S as the input spin configuration
#         S = S0
#
#         for i in range(1,self.num_layers+1):
#             # Compute effective field layer Fi
#             Fi = self.effective_field_layers[i-1](S)
#
#             # Compute quasi-particle layer Si
#             Si = torch.tanh(self.quasi_particle_layers[i-1](S0)) * Fi
#
#             # Update S for the next layer
#             S = Si
#
#         # Output layer to compute energy
#         E = self.output_layer(S).sum(dim=1, keepdim=True)  # Ensure scalar output per sample
#
#         return E


# # System Parameters
# L = 15  # Number of spins
# r = 2   # Number of spins in each interaction term

data_inDir=f"./data_inf_range_model_L{L}_r{r}/"
fileNameTrain=data_inDir+"/inf_range.train.pkl"

# Load data from pickle
with open(fileNameTrain, 'rb') as f:
    X_train, Y_train = pickle.load(f)


num_sample,num_spins=X_train.shape

# Hyperparameters
batch_size = 64
learning_rate = 0.001
weight_decay = 0.01  # L2 regularization strength
num_epochs = int(num_sample/batch_size*3)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device="+str(device))

# Load data
X_train = torch.tensor(X_train, dtype=torch.float64).to(device)  # Move to device
Y_train = torch.tensor(Y_train, dtype=torch.float64).view(-1, 1).to(device)

# Define a custom dataset
# class CustomDataset(Dataset):
#     def __init__(self, X, Y):
#         self.X = X
#         self.Y = Y
#
#     def __len__(self):
#         return len(self.X)
#
#     def __getitem__(self, idx):
#         return self.X[idx], self.Y[idx]


# Create Dataset and DataLoader
dataset = CustomDataset(X_train, Y_train)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# num_layers = 5  # Number of DSNN layers
num_neurons = int(num_spins*2)  # Number of neurons per layer
model = DSNN(num_spins=num_spins, num_layers=num_layers, num_neurons=num_neurons).double().to(device)


# Define loss function and optimizer with L2 regularization
# Optimizer and loss
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Define a step learning rate scheduler
# Reduce learning rate by a factor of gamma every step_size epochs
# decrease_over=70
scheduler = StepLR(optimizer, step_size=decrease_over, gamma=0.7)


out_model_Dir=f"./out_model_L{L}_r{r}_layer{num_layers}/"
Path(out_model_Dir).mkdir(exist_ok=True,parents=True)
loss_file_content=[]

print(f"num_spin={num_spins}")
print(f"num_neurons={num_neurons}")
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
        epoch_loss += loss.item() * X_batch.size(0)

    # Average loss over total samples
    average_loss = epoch_loss / len(dataset)

    # Print and log epoch summary
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}")
    loss_file_content.append(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.8f}\n")

    # Update the learning rate
    scheduler.step()

    # Optionally print the current learning rate
    current_lr = scheduler.get_last_lr()[0]
    print(f"Learning Rate after Epoch {epoch + 1}: {current_lr:.8e}")

# Save the loss log
with open(out_model_Dir + "/training_log.txt", "w+") as fptr:
    fptr.writelines(loss_file_content)

# Save the model
torch.save(model.state_dict(), out_model_Dir + "/model.pth")
tTrainEnd = datetime.now()

print("Training time:", tTrainEnd - tTrainStart)