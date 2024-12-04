import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from decimal import Decimal, getcontext
import numpy as np
from datetime import datetime

from torch.utils.data import Dataset, DataLoader

from model_dsnn_config import format_using_decimal, DSNN,CustomDataset

from model_dsnn_config import num_layers,num_neurons,L,r,device



data_inDir=f"./data_inf_range_model_L{L}_r{r}/"
fileNameTest=data_inDir+"/inf_range.test.pkl"

model_inDir=f"./out_model_L{L}_r{r}_layer{num_layers}/"

model_file=model_inDir+"/DSNN_model.pth"

with open(fileNameTest, 'rb') as f:
    X_test, Y_test = pickle.load(f)

# Convert to PyTorch tensors
X_test = torch.tensor(X_test, dtype=torch.float64).to(device)
Y_test = torch.tensor(Y_test, dtype=torch.float64).view(-1, 1).to(device)

# Set model to evaluation mode
model = DSNN(num_spins=L, num_layers=num_layers, num_neurons=num_neurons).double()
criterion = nn.MSELoss()
model.load_state_dict(torch.load(model_file))

# Move the model to the appropriate device
model = model.to(device)

# Set the model to evaluation mode
model.eval()

print("Model successfully loaded and ready for evaluation.")

# Disable gradient computation for evaluation
errors = []
with torch.no_grad():
    # Forward pass to get predictions
    predictions = model(X_test)

    # Compute loss or other evaluation metrics
    test_loss = criterion(predictions, Y_test).item()
    batch_errors = (predictions - Y_test).cpu().numpy()  # Convert to NumPy for easier handling
    errors.extend(batch_errors.flatten())  # Flatten and add to the list

# print(errors)
print(f"Test Loss: {test_loss:.4f}")
# Convert errors to a NumPy array
errors = np.array(errors)

# Compute the variance of the errors
error_variance = np.var(errors)
print(f"Error Variance: {error_variance:.8f}")