import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from decimal import Decimal, getcontext
import numpy as np
from datetime import datetime

from torch.utils.data import Dataset, DataLoader

from model_dsnn_train import format_using_decimal, DSNN,CustomDataset

from model_dsnn_train import TStr,num_layers,num_neurons,device,num_spins

inDir=f"./data_rand_energy_spin_T_{TStr}/"

fileNameTest=inDir+"/rand_energy_spin.test.pkl"
model_file_name = inDir + "/model.pth"
with open(fileNameTest, 'rb') as f:
    X_test, Y_test = pickle.load(f)

# Convert to PyTorch tensors
X_test = torch.tensor(X_test, dtype=torch.float64)  # Input features
Y_test = torch.tensor(Y_test, dtype=torch.float64).view(-1, 1)  # Labels reshaped to (num_samples, 1)
# Create DataLoader for test data
test_dataset = CustomDataset(X_test, Y_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = DSNN(num_spins=num_spins, num_layers=num_layers, num_neurons=num_neurons).double()

# Load the trained model parameters
model.load_state_dict(torch.load(model_file_name))
model.eval()  # Set the model to evaluation mode

# Evaluate the model
criterion = torch.nn.MSELoss()
test_loss = 0.0

with torch.no_grad():  # Disable gradient computation
    for X_batch, Y_batch in test_loader:
        # Forward pass
        predictions = model(X_batch)

        # Compute loss
        loss = criterion(predictions, Y_batch)
        test_loss += loss.item() * X_batch.size(0)  # Accumulate scaled loss


# Compute average loss over the test set
average_test_loss = test_loss / len(test_loader.dataset)

print(f"Test Loss (MSE): {average_test_loss:.4f}")