from model_qt_dsnn_config import *





def set_random_seed(seed):
    """
    Sets the random seed for reproducibility across PyTorch, NumPy, and Python's random module.

    Args:
        seed (int): The random seed value.
    """
    # Set the random seed for Python's random module
    # random.seed(seed)

    # Set the random seed for NumPy
    np.random.seed(seed)

    # Set the random seed for PyTorch
    torch.manual_seed(seed)

    # If using a GPU, ensure reproducibility for CUDA operations
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If using multi-GPU

    # Ensure deterministic behavior for CUDA (slightly reduces performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



# set_random_seed(18)


U = torch.tensor([[1.0, 0.0, 0.0],
                  [0.0, 0.0, -1.0],
                  [0.0, 1.0, 0.0]], dtype=torch.float)  # Example 3x3 orthogonal matrix


# Apply U to each vector in X_train_tensor
def apply_orthogonal_matrix(X_train_tensor, U):
    """
    Applies the orthogonal matrix U to each vector on the N x N grid in every sample.

    Args:
        X_train_tensor (torch.Tensor): Input tensor of shape (num_samples, 3, N, N).
        U (torch.Tensor): Orthogonal matrix of shape (3, 3).

    Returns:
        torch.Tensor: Transformed tensor with the same shape as the input.
    """
    # Ensure U has the correct shape (3, 3)
    assert U.shape == (3, 3), "U must be a 3x3 orthogonal matrix."

    # Permute X_train_tensor to shape (num_samples, N, N, 3) for easier broadcasting
    X_train_permuted = X_train_tensor.permute(0, 2, 3, 1)  # Shape: (num_samples, N, N, 3)

    # Apply matrix multiplication
    # Use torch.matmul for batch matrix-vector multiplication
    transformed = torch.matmul(X_train_permuted, U.T)  # Shape: (num_samples, N, N, 3)

    # Permute back to original shape (num_samples, 3, N, N)
    transformed = transformed.permute(0, 3, 1, 2)  # Shape: (num_samples, 3, N, N)

    return transformed




# Instantiate the network
model = dsnn_qt(
    input_channels=3,
    phi0_out_channels=C,
    T_out_channels=C,
    nonlinear_conv1_out_channels=C,
    nonlinear_conv2_out_channels=C,
    final_out_channels=1,
    filter_size=filter_size
).to(device)


inDir=f"./train_test_data/N{N}/"

in_pkl_train_file=inDir+"/db.train.pkl"

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

# step_num_after_S1=3
num_epochs = 500
learning_rate = 1e-3
weight_decay = 1e-5
# decrease_over = 50
# decrease_rate = 0.6


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

        # Initialize S1 for the batch
        S1 = model.initialize_S1(X_batch)

        # Forward pass
        predictions = model(X_batch, S1, steps=step_num_after_S1)

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

suffix_str=f"_over{decrease_overStr}_rate{decrease_rateStr}"
# Save training log to file
with open(out_model_dir+f"/training_log{suffix_str}.txt", "w") as f:
    f.writelines(loss_file_content)

# Save the trained model
torch.save(model.state_dict(), out_model_dir+f"/dsnn_qt_trained{suffix_str}.pth")
print("Training complete. Model saved as 'dsnn_qt_trained.pth'.")

tEnd=datetime.now()

print(f"time: {tEnd-tStart}")