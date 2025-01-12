from model_qt_dsnn_config import *
import sys

# step_num_after_S1=3
argErrCode=3
if (len(sys.argv)!=7):
    print("wrong number of arguments")
    print("example: python train_qt_dqnn.py num_epochs decrease_over decrease_rate step_num_after_S1 C N")
    exit(argErrCode)
num_epochs = int(sys.argv[1])
learning_rate = 1e-3
weight_decay = 1e-5
decrease_over = int(sys.argv[2])

decrease_rate = float(sys.argv[3])

step_num_after_S1=int(sys.argv[4])

C=int(sys.argv[5])

N=int(sys.argv[6])

inDir=f"./train_test_data/N{N}/"
num_samples_in=200000
in_pkl_train_file=inDir+f"/db.train_num_samples{num_samples_in}.pkl"

with open(in_pkl_train_file,"rb") as fptr:
    X_train, Y_train=pickle.load(fptr)


# Convert to NumPy arrays

X_train_array = np.array(X_train)  # Shape: (num_samples, 3,N, N, )
Y_train_array = np.array(Y_train)  # Shape: (num_samples,)

# Convert NumPy arrays to PyTorch tensors with dtype=torch.float64
X_train_tensor = torch.tensor(X_train_array, dtype=torch.float)  # Shape: (num_samples, 3, N, N)
Y_train_tensor = torch.tensor(Y_train_array, dtype=torch.float)  # Shape: (num_samples,)


# Instantiate the dataset
train_dataset = CustomDataset(X_train_tensor, Y_train_tensor)

# Create DataLoader for training
batch_size = 1000 # Define batch size
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Instantiate the network
model=DSNN_qt(
symmetrization_out_channels=1,
symmetrization_kernel_size=filter_size,
stepsAfterInit=step_num_after_S1,
nonlinear_conv1_out_channels=C,
nonlinear_conv_kernel_size=filter_size,
nonlinear_conv_dropout_prob=0.1,
nonlinear_conv_linear_out_features1=2*N*N,
nonlinear_conv_linear_out_features2=N*N,
no_conv_lin_layer1_in_dim=N*N,
no_conv_lin_layer1_out_dim=N*N,
no_conv_lin_layer2_out_dim=N*N
).to(device)
# Optimizer, scheduler, and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = StepLR(optimizer, step_size=decrease_over, gamma=decrease_rate)
criterion = nn.MSELoss()
tStart=datetime.now()
# To log loss values for each epoch
loss_file_content = []
print(f"device={device}")
# Training loop
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    epoch_loss = 0  # Reset epoch loss
    for X_batch, Y_batch in train_loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)  # Move batch to device
        Y_batch = Y_batch.unsqueeze(1)
        # print(f"Y_batch.shape={Y_batch.shape}")
        # Initialize S1 for the batch
        S1 = model.initialize_S1(X_batch)
        # Forward pass
        predictions = model(X_batch, S1)
        # Compute loss
        loss = criterion(predictions, Y_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Accumulate batch loss (scaled by batch size)
        epoch_loss += loss.item() * X_batch.size(0)
    # Compute average loss over all samples
    average_loss = epoch_loss / len(train_dataset)
    # Log epoch summary
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}")
    loss_file_content.append(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.8f}\n")
    # Update the learning rate
    scheduler.step()
    # Optionally log the current learning rate
    current_lr = scheduler.get_last_lr()[0]
    print(f"Learning Rate after Epoch {epoch + 1}: {current_lr:.8e}")


out_model_dir=f"./out_model_data/N{N}/C{C}/layer{step_num_after_S1}/"
Path(out_model_dir).mkdir(exist_ok=True,parents=True)
decrease_overStr=format_using_decimal(decrease_over)
decrease_rateStr=format_using_decimal(decrease_rate)
suffix_str=f"_over{decrease_overStr}_rate{decrease_rateStr}_epoch{num_epochs}_num_samples{num_samples_in}"
# Save training log to file
with open(out_model_dir+f"/training_log{suffix_str}.txt", "w") as f:
    f.writelines(loss_file_content)

# Save the trained model
torch.save(model.state_dict(), out_model_dir+f"/dsnn_qt_trained{suffix_str}.pth")
print("Training complete")

tEnd=datetime.now()

print(f"time: {tEnd-tStart}")
