from model_dsnn_config import *

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

num_layers =3
K=455
data_inDir=f"./data_inf_range_model_L{L}_K_{K}_r{r}/"
fileNameTest=data_inDir+"/inf_range.test.pkl"

model_inDir=f"./out_model_L{L}_K{K}_r{r}_layer{num_layers}/"

model_file=model_inDir+"/DNN_model.pth"

with open(fileNameTest, 'rb') as f:
    X_test, Y_test = pickle.load(f)

# Convert to PyTorch tensors
X_test = torch.tensor(X_test, dtype=torch.float64).to(device)
Y_test = torch.tensor(Y_test, dtype=torch.float64).view(-1, 1).to(device)
print(f"Y_test.shape={Y_test.shape}")
model = DNN(num_spins=L, num_layers=num_layers, num_neurons=num_neurons).double().to(device)


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
# error_variance = np.var(errors)
# print(f"Error Variance: {error_variance:.8f}")
std_loss=np.sqrt(test_loss)

outTxtFile=model_inDir+f"test_DNN.txt"

out_content=f"MSE_loss={format_using_decimal(test_loss)}, std_loss={format_using_decimal(std_loss)}\n"
with open(outTxtFile,"w+") as fptr:
    fptr.write(out_content)