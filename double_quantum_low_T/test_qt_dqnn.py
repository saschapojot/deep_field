from model_qt_dsnn_config import *


# Evaluation Function
def evaluate_model(model, test_loader, device):
    """
    Evaluate the trained model on the test dataset.

    Args:
        model (nn.Module): Trained model.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to run evaluation on.

    Returns:
        float: Mean squared error on the test dataset.
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    criterion = torch.nn.MSELoss()  # Loss function

    with torch.no_grad():  # No need to compute gradients during evaluation
        for X_batch, Y_batch in test_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)  # Move data to device

            # Initialize S1 for the batch
            S1 = model.initialize_S1(X_batch)

            # Forward pass
            predictions = model(X_batch, S1, steps=step_num_after_S1)

            # Compute loss
            loss = criterion(predictions, Y_batch)
            total_loss += loss.item() * X_batch.size(0)  # Accumulate loss weighted by batch size

    # Compute average loss
    average_loss = total_loss / len(test_loader.dataset)
    return average_loss

N=10
C=25
#layer
step_num_after_S1=0

decrease_over = 50

decrease_rate = 0.95


num_epochs = 1000

numSample=400000
decrease_overStr=format_using_decimal(decrease_over)
decrease_rateStr=format_using_decimal(decrease_rate)

suffix_str=f"_over{decrease_overStr}_rate{decrease_rateStr}_epoch{num_epochs}_num_samples{numSample}"
in_model_dir=f"./out_model_data/N{N}/C{C}/layer{step_num_after_S1}/"

in_model_file=in_model_dir+f"dsnn_qt_trained_over{decrease_overStr}_rate{decrease_rateStr}_epoch{num_epochs}_num_samples{numSample}.pth"
inDir=f"./train_test_data/N{N}/"

in_pkl_test_file=inDir+f"/db.test_num_samples{numSample}.pkl"

with open(in_pkl_test_file,"rb") as fptr:
    X_test, Y_test = pickle.load(fptr)

X_test_array=np.array(X_test)
Y_test_array=np.array(Y_test)
print(f"Y_test_array.shape={Y_test_array.shape}")

X_test_tensor=torch.tensor(X_test_array,dtype=torch.float)

Y_test_tensor=torch.tensor(Y_test_array,dtype=torch.float)

batch_size = 1000  # Define batch size
# Create test dataset and DataLoader
test_dataset = CustomDataset(X_test_tensor, Y_test_tensor)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)  # Batch size can be adjusted

# Load the trained model
model = dsnn_qt(
        input_channels=3,
        phi0_out_channels=C,
        T_out_channels=C,
        nonlinear_conv1_out_channels=C,
        nonlinear_conv2_out_channels=C,
        final_out_channels=1,
        filter_size=filter_size
    )


model.load_state_dict(torch.load(in_model_file, map_location=device))  # Load saved weights
model.to(device)  # Move model to device

# Evaluate the model
test_loss = evaluate_model(model, test_loader, device)
std_loss=np.sqrt(test_loss)
print(f"Test Loss (MSE): {test_loss:.6f}")

outTxtFile=in_model_dir+f"/test_over{decrease_overStr}_rate{decrease_rateStr}_epoch{num_epochs}_num_samples{numSample}.txt"

out_content=f"MSE_loss={format_using_decimal(test_loss)}, std_loss={format_using_decimal(std_loss)}  N={N}, C={C}, layer={step_num_after_S1}, decrease_over={decrease_overStr}, decrease_rate={decrease_rateStr}, num_epochs={num_epochs}"

with open(outTxtFile,"w+") as fptr:
    fptr.write(out_content)