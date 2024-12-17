from model_dsnn_config import L,r,decrease_over,num_layers,format_using_decimal
from model_dsnn_config import num_neurons,decrease_rate,batch_size,learning_rate,weight_decay

from model_dsnn_config import  epoch_multiple,CustomDataset
import torch
import torch.nn as nn
import pickle
import numpy as np
from datetime import datetime
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from pathlib import Path



class DNN(nn.Module):
    def __init__(self, num_spins, num_layers, num_neurons):
        super(DNN,self).__init__()

        self.num_layers = num_layers

        # Effective field layers (F_i)
        self.effective_field_layers = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(num_spins if i == 1 else num_neurons, num_neurons),
                nn.Tanh(),
                nn.Linear(num_neurons, num_neurons)
            ) for i in range(1, num_layers + 1)]
        )

        # Quasi-particle layers (S_i)
        self.quasi_particle_layers = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(num_neurons, num_neurons),
                nn.Tanh(),
                nn.Linear(num_neurons, num_neurons)
            ) for i in range(1, num_layers + 1)]
        )

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(num_neurons, num_neurons),  # Optional intermediate transformation
            nn.Tanh(),
            nn.Linear(num_neurons, 1)  # Final mapping to scalar
        )

    def forward(self, S0):
        S = S0
        for i in range(1, self.num_layers + 1):
            Fi = self.effective_field_layers[i - 1](S)
            Si = self.quasi_particle_layers[i - 1](Fi)
            S = Si

        # Output layer to compute energy
        E = self.output_layer(S).sum(dim=1, keepdim=True)  # Ensure scalar output per sample

        return E


data_inDir=f"./data_inf_range_model_L{L}_r{r}/"
fileNameTest=data_inDir+"/inf_range.test.pkl"

with open(fileNameTest,"rb") as fptr:
    X_test, Y_test = pickle.load(fptr)

X_test=np.array(X_test)
Y_test=np.array(Y_test)
# Step 3: Prepare Data for Testing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

X_test = torch.tensor(X_test, dtype=torch.float64).to(device)

Y_test = torch.tensor(Y_test, dtype=torch.float64).view(-1, 1).to(device)

# Use DataLoader for batch processing
test_dataset = CustomDataset(X_test, Y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Step 4: Load the Saved Model
out_model_Dir = f"./out_model_L{L}_r{r}_layer{num_layers}/"
model_path = out_model_Dir + "/DNN_model.pth"

model = DNN(num_spins=L, num_layers=num_layers, num_neurons=num_neurons).double()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)  # Move model to the same device
model.eval()

print("Model successfully loaded and set to evaluation mode.")

# Step 5: Test the Model and Compute Loss
criterion = nn.MSELoss()
test_loss = 0
total_samples = 0

all_predictions = []
all_true_values = []

# Disable gradient computation for evaluation
errors = []

with torch.no_grad():
    for X_batch, Y_batch in test_loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)  # Ensure batches are on the same device

        # Forward pass to get predictions
        predictions = model(X_batch)

        # Compute batch loss
        loss = criterion(predictions, Y_batch)
        test_loss += loss.item() * X_batch.size(0)
        total_samples += X_batch.size(0)

        # Store predictions and true values
        all_predictions.append(predictions.cpu().numpy())
        all_true_values.append(Y_batch.cpu().numpy())

        # Compute errors
        batch_errors = (predictions - Y_batch).cpu().numpy()
        errors.extend(batch_errors.flatten())

# Compute average test loss
average_test_loss = test_loss / total_samples
print(f"Test Loss: {average_test_loss:.4f}")

std_loss=np.sqrt(average_test_loss)
outTxtFile=out_model_Dir+f"test_DNN.txt"

out_content=f"MSE_loss={format_using_decimal(average_test_loss)}, std_loss={format_using_decimal(std_loss)}\n"
with open(outTxtFile,"w+") as fptr:
    fptr.write(out_content)