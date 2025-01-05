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


if (len(sys.argv) != 3):
    print("wrong number of arguments.")
    exit(21)


num_layers = int(sys.argv[1])
K=int(sys.argv[2])
data_inDir=f"./data_inf_range_model_L{L}_K_{K}_r{r}/"
fileNameTrain=data_inDir+"/inf_range.train.pkl"

with open(fileNameTrain,"rb") as fptr:
    X_train, Y_train = pickle.load(fptr)


num_sample,num_spins=X_train.shape
print(f"num_sample={num_sample}")
num_epochs =int(num_sample/batch_size*epoch_multiple)
print(f"num_epochs={num_epochs}")
X_train = torch.tensor(X_train, dtype=torch.float64).to(device)  # Move to device
Y_train = torch.tensor(Y_train, dtype=torch.float64).view(-1, 1).to(device)

# Create Dataset and DataLoader
dataset = CustomDataset(X_train, Y_train)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


model = DNN(num_spins=num_spins, num_layers=num_layers, num_neurons=num_neurons).double().to(device)

# Define loss function and optimizer with L2 regularization
# Optimizer and loss
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)



# Define a step learning rate scheduler
# Reduce learning rate by a factor of gamma every step_size epochs
# decrease_over=70
scheduler = StepLR(optimizer, step_size=decrease_over, gamma=decrease_rate)

out_model_Dir=f"./out_model_L{L}_K{K}_r{r}_layer{num_layers}/"
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
with open(out_model_Dir + "/DNN_training_log.txt", "w+") as fptr:
    fptr.writelines(loss_file_content)

# Save the model
torch.save(model.state_dict(), out_model_Dir + "/DNN_model.pth")
tTrainEnd = datetime.now()

print("Training time:", tTrainEnd - tTrainStart)