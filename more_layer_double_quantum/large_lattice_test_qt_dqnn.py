from model_qt_dsnn_config import *

#this function loads test data for larger lattices
# def diff_2_ratio(E_pred,E_true):
#     return (E_pred-E_true)**2/E_true**2
#test with mse loss and custom loss
# Evaluation Function
def evaluate_model(N,model, test_loader, device):
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
    all_predictions = []  # To store all predictions
    custom_metric_sum = 0.0
    custom_metric_sum_another=0
    with torch.no_grad():  # No need to compute gradients during evaluation
        for X_batch, Y_batch in test_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)  # Move data to device

            # Initialize S1 for the batch
            S1 = model.initialize_S1(X_batch)

            # Forward pass
            predictions = model(X_batch, S1)
            # print(f"len(X_batch)={len(X_batch)}, len(Y_batch)={len(Y_batch)}")
            # print(f"Y_batch size: {Y_batch.shape}")
            # print(f"predictions.shape={predictions.shape}")
            all_predictions.append(predictions.cpu())
            # predictions_tilde=(predictions)/N**delta
            # Y_batch_tilde=(Y_batch)/N**beta
            # batch_custom_metric = ((predictions_tilde - Y_batch_tilde) ** 2 / (Y_batch_tilde ** 2)).sum().item()
            # batch_custom_metric_another = ((predictions_tilde - Y_batch_tilde) **2 /predictions_tilde**2).sum().item()
            # custom_metric_sum += batch_custom_metric
            # custom_metric_sum_another+=batch_custom_metric_another


            # Compute loss
            loss = criterion(predictions, Y_batch)
            total_loss += loss.item() * X_batch.size(0)  # Accumulate loss weighted by batch size

    # Compute average loss
    average_loss = total_loss / len(test_loader.dataset)
    all_predictions = torch.cat(all_predictions, dim=0)
    return average_loss,all_predictions#,custom_metric_sum,custom_metric_sum_another
N_for_model=10
N=35
C=10
# beta=1.9999403561992048
# delta=2.026271899527667
# c_true=-0.0014459645634784324
# c_pred=-2.8370278369596145
#layer
step_num_after_S1=0

decrease_over = 50

decrease_rate = 0.9


num_epochs = 1200

decrease_overStr=format_using_decimal(decrease_over)
decrease_rateStr=format_using_decimal(decrease_rate)

# suffix_str=f"_over{decrease_overStr}_rate{decrease_rateStr}_epoch{num_epochs}_num_samples200000"
in_model_dir=f"./out_model_data_noCheckPoint/N{N_for_model}/C{C}/layer{step_num_after_S1}/"

in_model_file=in_model_dir+f"dsnn_qt_trained_over{decrease_overStr}_rate{decrease_rateStr}_epoch{num_epochs}_num_samples200000.pth"
inDir=f"./larger_lattice_test_performance/N{N}/C{C}/layer{step_num_after_S1}/"

num_suffix=40000
in_pkl_test_file=inDir+f"/db.test_num_samples{num_suffix}.pkl"

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
        filter_size=filter_size,
stepsAfterInit=step_num_after_S1
    )


model.load_state_dict(torch.load(in_model_file, map_location=device))  # Load saved weights
model.to(device)  # Move model to device

# Evaluate the model
test_loss,predictions = evaluate_model(N,model, test_loader, device)
std_loss=np.sqrt(test_loss)
# print(predictions)
print(f"Test Loss (MSE): {test_loss:.6f}")
# print(f"custom_metric_sum={custom_metric_sum:.6f}")
# custom_err=np.sqrt(custom_metric_sum/len(predictions))
# print(f"custom_err={custom_err:.6f}")
#
# print(f"custom_metric_sum_another={custom_metric_sum_another:.6f}")
# custom_err_another=np.sqrt(custom_metric_sum_another/len(predictions))
# print(f"custom_err_another={custom_err_another:.6f}")
predictions=np.array(predictions)
# print(f"len(predictions)={len(predictions)}")
pred_mean=np.mean(predictions)
pred_std = np.std(predictions)

outResultDir=f"./larger_lattice_test_performance/N{N}/C{C}/layer{step_num_after_S1}/"
Path(outResultDir).mkdir(parents=True, exist_ok=True)

outTxtFile=outResultDir+f"/test_over{decrease_overStr}_rate{decrease_rateStr}_epoch{num_epochs}_num_samples{num_suffix}.txt"

out_content=f"MSE_loss={format_using_decimal(test_loss)}, std_loss={format_using_decimal(std_loss)}  N={N}, C={C}, layer={step_num_after_S1}, decrease_over={decrease_overStr}, decrease_rate={decrease_rateStr}, num_epochs={num_epochs}, pred_mean={pred_mean}, pred_std={pred_std}"

with open(outTxtFile,"w+") as fptr:
    fptr.write(out_content)