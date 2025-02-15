import os
import re
from model_qt_dsnn_config import *
import glob


#this script compute test performance using modeled saved at different epochs
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
            predictions = model(X_batch, S1)

            # Compute loss
            loss = criterion(predictions, Y_batch)
            total_loss += loss.item() * X_batch.size(0)  # Accumulate loss weighted by batch size

    # Compute average loss
    average_loss = total_loss / len(test_loader.dataset)
    return average_loss


N=10
C=30
#layer
step_num_after_S1=2

decrease_over = 50

decrease_rate = 0.9
# num_epochs = 50

decrease_overStr=format_using_decimal(decrease_over)
decrease_rateStr=format_using_decimal(decrease_rate)
in_model_dir=f"./out_model_data/N{N}/C{C}/layer{step_num_after_S1}/"
pth_file_vec=[]
for pthFile in glob.glob(os.path.join(in_model_dir, "*.pth")):
    pth_file_vec.append(pthFile)

in_pkl_Dir=f"./train_test_data/N{N}/"
in_pkl_test_file=in_pkl_Dir+"/db.test_num_samples200000.pkl"
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



pattern_epoch=r'epoch(\d+)'
def extract_epoch(pth_file):
    match_epoch=re.search(pattern_epoch,pth_file)
    if match_epoch:
        return int(match_epoch.group(1))
    else:
        print(f"format error: {pth_file}")
        exit(12)
epoch_vals=[extract_epoch(file) for file in pth_file_vec]

inds_epoch_vals=np.argsort(epoch_vals)
sorted_pth_file_vec=[pth_file_vec[j] for j in inds_epoch_vals]
sorted_epoch_vec=[epoch_vals[j] for j in inds_epoch_vals]

def one_epoch_test(in_pth_file,test_loader, device):
    # Load the trained model
    checkpoint = torch.load(in_pth_file, map_location=device)
    model = dsnn_qt(
        input_channels=3,
        phi0_out_channels=C,
        T_out_channels=C,
        nonlinear_conv1_out_channels=C,
        nonlinear_conv2_out_channels=C,
        final_out_channels=1,
        filter_size=filter_size,
        stepsAfterInit=step_num_after_S1
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)  # Move model to device
    test_loss = evaluate_model(model, test_loader, device)
    std_loss = np.sqrt(test_loss)
    return test_loss,std_loss


def one_line_outFile(epochNum,test_loss,std_loss,N,C,layer,decrease_overStr,decrease_rateStr):
    msg=f"num_epochs={epochNum}, MSE_loss={format_using_decimal(test_loss)}, std_loss={format_using_decimal(std_loss)}  N={N}, C={C}, layer={layer}, decrease_over={decrease_overStr}, decrease_rate={decrease_rateStr}\n"
    return msg


#contents in output file
outLineAll=[]
outTextFileName=in_model_dir+"/test_over_epochs.txt"
for j in range(0,len(sorted_pth_file_vec)):

    epoch=sorted_epoch_vec[j]
    print(f"epoch={epoch}")
    pth_file=sorted_pth_file_vec[j]
    test_loss, std_loss=one_epoch_test(pth_file,test_loader,device)
    oneLine=one_line_outFile(epoch,test_loss,std_loss,N,C,step_num_after_S1,decrease_over,decrease_rate)

    outLineAll.append(oneLine)

with open(outTextFileName,"w+") as fptr:
    fptr.writelines(outLineAll)