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
from model_dsnn_config import  num_neurons,decrease_rate,batch_size,learning_rate,weight_decay
from model_dsnn_config import  epoch_multiple,device


data_inDir=f"./data_inf_range_model_L{L}_r{r}/"
fileNameTrain=data_inDir+"/inf_range.train.pkl"

# Load data from pickle
with open(fileNameTrain, 'rb') as f:
    X_train, Y_train = pickle.load(f)


num_sample,num_spins=X_train.shape

# Hyperparameters
# batch_size = 64
# learning_rate = 0.001
# weight_decay = 0.01  # L2 regularization strength
num_epochs = int(num_sample/batch_size*epoch_multiple)

# Define device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("device="+str(device))

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
# num_neurons = int(num_spins*3)  # Number of neurons per layer
model = DSNN(num_spins=num_spins, num_layers=num_layers, num_neurons=num_neurons).double().to(device)


# Define loss function and optimizer with L2 regularization
# Optimizer and loss
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Define a step learning rate scheduler
# Reduce learning rate by a factor of gamma every step_size epochs
# decrease_over=70
scheduler = StepLR(optimizer, step_size=decrease_over, gamma=decrease_rate)


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
with open(out_model_Dir + "/DSNN_training_log.txt", "w+") as fptr:
    fptr.writelines(loss_file_content)

# Save the model
torch.save(model.state_dict(), out_model_Dir + "/DSNN_model.pth")
tTrainEnd = datetime.now()

print("Training time:", tTrainEnd - tTrainStart)