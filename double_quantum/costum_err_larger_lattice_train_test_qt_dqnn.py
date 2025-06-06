from model_qt_dsnn_config import *

def custom_loss(predictions, targets, regularization_strength=0.01):
    """
    Custom loss function that computes the weighted squared error divided by the target squared,
    with an additional L2 regularization term.

    Args:
        predictions (torch.Tensor): The model's predictions.
        targets (torch.Tensor): The ground truth targets.
        regularization_strength (float): The strength of the regularization term.

    Returns:
        torch.Tensor: The computed loss.
    """
    # Compute the squared error, normalized by targets^2
    error_term = torch.sum(((predictions - targets) ** 2) / (targets ** 2))

    # L2 regularization term (using the model parameters, or just predictions if preferred)
    regularization_term = torch.norm(predictions, p=2)

    # Combine the error term and regularization term
    total_loss = error_term + regularization_strength * regularization_term
    return total_loss


# Evaluation Function
def evaluate_model(model, test_loader, device,regularization_strength):
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
    criterion = custom_loss  # Loss function

    with torch.no_grad():  # No need to compute gradients during evaluation
        for X_batch, Y_batch in test_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)  # Move data to device

            # Initialize S1 for the batch
            S1 = model.initialize_S1(X_batch)

            # Forward pass
            predictions = model(X_batch, S1, steps=step_num_after_S1)

            # Compute loss
            loss = criterion(predictions, Y_batch,regularization_strength=regularization_strength)
            total_loss += loss.item()

    # Compute average loss
    average_loss = total_loss / len(test_loader.dataset)
    return average_loss
N_in_model=10
N=35
C=25
#layer
step_num_after_S1=0

decrease_over = 50

decrease_rate = 0.9

weight_decay = 1e-5
num_epochs = 1200

num_test_larger=40000
decrease_overStr=format_using_decimal(decrease_over)
decrease_rateStr=format_using_decimal(decrease_rate)

# suffix_str=f"_over{decrease_overStr}_rate{decrease_rateStr}_epoch{num_epochs}_num_samples200000"
in_model_dir=f"./custom_err_out_model_data/N{N_in_model}/C{C}/layer{step_num_after_S1}/"

in_model_file=in_model_dir+f"dsnn_qt_trained_over{decrease_overStr}_rate{decrease_rateStr}_epoch{num_epochs}_num_samples200000.pth"


in_pkl_dir=f"./custom_err_larger_lattice_test_performance/N{N}/C{C}/layer{step_num_after_S1}/"
# Path(in_pkl_dir).mkdir(parents=True, exist_ok=True)
in_pkl_test_file=in_pkl_dir+f"/db.test_num_samples{num_test_larger}.pkl"

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
test_loss = evaluate_model(model, test_loader, device,weight_decay)
std_loss=np.sqrt(test_loss)
print(f"Test Loss (MSE): {test_loss:.6f}")

outTxtFile=in_pkl_dir+f"/test_over{decrease_overStr}_rate{decrease_rateStr}_epoch{num_epochs}_num_samples{num_test_larger}.txt"

out_content=f"MSE_loss={format_using_decimal(test_loss)}, std_loss={format_using_decimal(std_loss)}  N={N}, C={C}, layer={step_num_after_S1}, decrease_over={decrease_overStr}, decrease_rate={decrease_rateStr}, num_epochs={num_epochs}"

with open(outTxtFile,"w+") as fptr:
    fptr.write(out_content)