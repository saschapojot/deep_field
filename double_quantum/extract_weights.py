import torch
import torch.nn as nn
from decimal import Decimal, getcontext
import pickle
from decimal import Decimal, getcontext
import numpy as np
from datetime import datetime
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys


#this script extract weights of the neural network


#defining the neural network, this part is copied from model_qt_dsnn_config.py, with some data written to file for visualization

N=10
C=25
step_num_after_S1=1
filter_size=5
device = torch.device("cpu")
outDir="./coefs/"
Path(outDir).mkdir(parents=True, exist_ok=True)
def format_using_decimal(value, precision=10):
    # Set the precision higher to ensure correct conversion
    getcontext().prec = precision + 2
    # Convert the float to a Decimal with exact precision
    decimal_value = Decimal(str(value))
    # Normalize to remove trailing zeros
    formatted_value = decimal_value.quantize(Decimal(1)) if decimal_value == decimal_value.to_integral() else decimal_value.normalize()
    return str(formatted_value)

class toFile_Phi0Layer(nn.Module):
    def __init__(self, out_channels, kernel_size, padding=2):
        """
        A modified Phi0Layer where the convolutional matrices (weights) are shared
        across the three input channels (Sigma_x, Sigma_y, Sigma_z).

        Args:
            out_channels (int): Number of output channels per input channel.
            kernel_size (int): Size of the convolution kernel.
            padding (int): Padding for the convolution operation. Default is 2.
        """
        super(toFile_Phi0Layer, self).__init__()

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        # Shared convolutional layers for W0 and W1
        self.shared_conv_W0 = nn.Conv2d(
            in_channels=1,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=False
        )

        self.shared_conv_W1 = nn.Conv2d(
            in_channels=1,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=False
        )

    def forward(self, x,save_path=None):
        """
        Forward pass for the Phi0Layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, N, N).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, N, N).
        """
        # Split input into three individual channels
        x_channels = torch.chunk(x, chunks=3, dim=1)  # [(batch_size, 1, N, N), ...]

        # Apply shared W0 and W1 convolutions to each channel
        conv_W0_outputs = [self.shared_conv_W0(channel) for channel in x_channels]
        conv_W1_outputs = [self.shared_conv_W1(channel) for channel in x_channels]

        # Perform element-wise multiplication for each channel
        multiplied_outputs = [
            conv_W0 * conv_W1 for conv_W0, conv_W1 in zip(conv_W0_outputs, conv_W1_outputs)
        ]  # [(batch_size, out_channels, N, N), ...]

        # Sum over the 3 input channels
        Phi0 = sum(multiplied_outputs)  # Shape: (batch_size, out_channels, N, N)
        if save_path:
            torch.save(Phi0, save_path)
            print(f"Phi0 saved to {save_path}")

        return Phi0

class toFile_TLayer(nn.Module):
    def __init__(self, out_channels, kernel_size, padding=2):
        """
        A layer with the same functionality as Phi0Layer:
        - Applies shared convolutional weights across three input channels.
        - Performs element-wise multiplication between convolution results.
        - Sums the results over the three input channels.

        Args:
            out_channels (int): Number of output channels per input channel.
            kernel_size (int): Size of the convolution kernel.
            padding (int): Padding for the convolution operation. Default is 2.
        """
        super(toFile_TLayer, self).__init__()

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        # Shared convolutional layers for W0 and W1
        self.shared_conv_W0 = nn.Conv2d(
            in_channels=1,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=False
        )

        self.shared_conv_W1 = nn.Conv2d(
            in_channels=1,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=False
        )

    def forward(self, x,save_path=None):
        """
        Forward pass for the TLayer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, N, N).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, N, N).
        """
        # Split input into three individual channels
        x_channels = torch.chunk(x, chunks=3, dim=1)  # [(batch_size, 1, N, N), ...]

        # Apply shared W0 and W1 convolutions to each channel
        conv_W0_outputs = [self.shared_conv_W0(channel) for channel in x_channels]
        conv_W1_outputs = [self.shared_conv_W1(channel) for channel in x_channels]

        # Perform element-wise multiplication for each channel
        multiplied_outputs = [
            conv_W0 * conv_W1 for conv_W0, conv_W1 in zip(conv_W0_outputs, conv_W1_outputs)
        ]  # [(batch_size, out_channels, N, N), ...]

        # Sum over the 3 input channels
        T_result = sum(multiplied_outputs)  # Shape: (batch_size, out_channels, N, N)
        if save_path:
            torch.save(T_result, save_path)
            print("T_result saved to {}".format(save_path))
        return T_result

class toFile_NonlinearLayer(nn.Module):
    def __init__(self, in_channels, conv1_out_channels, conv2_out_channels, kernel_size, padding=1):
        """
        Nonlinear Layer:
        1. A convolution using the input tensor.
        2. A tanh activation.
        3. Another convolution on the tanh output.
        it is used for f and g

        Args:
            in_channels (int): Number of input channels.
            conv1_out_channels (int): Number of output channels for the first convolution.
            conv2_out_channels (int): Number of output channels for the second convolution.
            kernel_size (int): Size of the convolution filter. Default is 3.
            padding (int): Padding for the convolution operation. Default is 1.
        """
        super(toFile_NonlinearLayer, self).__init__()

        # First convolution
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=conv1_out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )

        # Tanh activation
        self.tanh = nn.Tanh()

        # Second convolution
        self.conv2 = nn.Conv2d(
            in_channels=conv1_out_channels,
            out_channels=conv2_out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )

    def forward(self, x,save_path=None):
        """
        Forward pass through the Nonlinear Layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, N, N).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, conv2_out_channels, N, N).
        """
        # First convolution
        x = self.conv1(x)  # Shape: (batch_size, conv1_out_channels, N, N)

        # Apply tanh activation
        x = self.tanh(x)  # Shape: (batch_size, conv1_out_channels, N, N)

        # Second convolution
        x = self.conv2(x)  # Shape: (batch_size, conv2_out_channels, N, N)
        if save_path:
            torch.save(x, save_path)
            print("x saved to {}".format(save_path))
        return x


class toFile_dsnn_qt(nn.Module):
    def __init__(self, input_channels,
                 phi0_out_channels, T_out_channels,
                 nonlinear_conv1_out_channels, nonlinear_conv2_out_channels,
                 final_out_channels=1,  # For N x N matrix
                 filter_size=5):
        """
        Neural network that computes S_{n+1} and maps it to a scalar E.

        Args:
            input_channels (int): Number of input channels (e.g., 3 for x, y, z components).
            phi0_out_channels (int): Number of output channels from Phi0Layer.
            T_out_channels (int): Number of output channels from TLayer.
            nonlinear_conv1_out_channels (int): Number of output channels for the first convolution in NonlinearLayer.
            nonlinear_conv2_out_channels (int): Number of output channels for the second convolution in NonlinearLayer.
            final_out_channels (int): Output channels for the final mapping (should be 1 for N x N matrix).
            filter_size (int): Size of the convolution filter in Phi0Layer and TLayer.
        """
        super(toFile_dsnn_qt, self).__init__()
        # Phi0Layer
        self.phi0_layer = toFile_Phi0Layer(out_channels=phi0_out_channels, kernel_size=filter_size)
        # TLayer
        self.T_layer = toFile_TLayer(out_channels=T_out_channels, kernel_size=filter_size)

        # NonlinearLayer for S_n and F_n+1
        self.nonlinear_layer = toFile_NonlinearLayer(
            in_channels=nonlinear_conv2_out_channels,
            conv1_out_channels=nonlinear_conv1_out_channels,
            conv2_out_channels=nonlinear_conv2_out_channels,
            kernel_size=3,  # Default convolution kernel size
            padding=1
        )

        # NonlinearLayer for processing input after TLayer
        self.nonlinear_layer_input = toFile_NonlinearLayer(
            in_channels=T_out_channels,
            conv1_out_channels=nonlinear_conv1_out_channels,
            conv2_out_channels=nonlinear_conv2_out_channels,
            kernel_size=3,
            padding=1
        )

        # Final mapping layer to N x N matrix
        self.final_mapping_layer = toFile_NonlinearLayer(
            in_channels=nonlinear_conv2_out_channels,
            conv1_out_channels=nonlinear_conv2_out_channels,
            conv2_out_channels=final_out_channels,  # Map to 1 channel for N x N matrix
            kernel_size=3,
            padding=1
        )

    def initialize_S1(self, x):
        """
        Initializes S1 using the input x.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, N, N).

        Returns:
            torch.Tensor: The tensor S1.
        """
        # Step 1: Compute F1 from Phi0Layer and NonlinearLayer
        phi0_output = self.phi0_layer(x)
        out_Phi0File=outDir+"/Phi0.pth"
        torch.save(phi0_output, out_Phi0File)
        print(f"Phi0 saved to {out_Phi0File}")
        F1 = self.nonlinear_layer(phi0_output)
        out_F1File=outDir+"/F1.pth"
        torch.save(F1, out_F1File)
        print("F1 saved to {}".format(out_F1File))
        # Step 2: Pass input through TLayer and NonlinearLayer
        T_output = self.T_layer(x)#T1
        out_T1File=outDir+"/T1.pth"
        torch.save(T_output, out_T1File)

        print("T1 saved to {}".format(out_T1File))
        nonlinear_output = self.nonlinear_layer_input(T_output)

        # Step 3: Compute S1 as pointwise multiplication of F1 and nonlinear_output
        S1 = F1 * nonlinear_output
        out_S1File=outDir+"/S1.pth"
        torch.save(S1, out_S1File)
        print("S1 saved to {}".format(out_S1File))

        return S1

    def forward(self, x, Sn, steps=1):
        """
        Forward pass through the recursive neural network to compute the target E.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, N, N).
            Sn (torch.Tensor): The tensor S_n of shape (batch_size, nonlinear_conv2_out_channels, N, N).
            steps (int): Number of recursive steps to compute S_{n+1}.

        Returns:
            torch.Tensor: The target scalar E for each batch.
        """
        for j in range(steps):
            # Step 1: Compute F_{n+1} by passing S_n through NonlinearLayer
            Fn_plus_1 = self.nonlinear_layer(Sn)
            ind=j+2
            out_F_file=outDir+f"/F{ind}.pth"
            torch.save(Fn_plus_1, out_F_file)
            print(f"F_{ind} saved to {out_F_file}")
            # Step 2: Pass input through TLayer and NonlinearLayer
            T_output = self.T_layer(x)
            out_T_file=outDir+f"/T_{ind}.pth"
            torch.save(T_output, out_T_file)
            print(f"T{ind} saved to {out_T_file}")
            nonlinear_output = self.nonlinear_layer_input(T_output)

            # Step 3: Compute S_{n+1} as pointwise multiplication of Fn_plus_1 and nonlinear_output
            Sn = Fn_plus_1 * nonlinear_output
            out_S_file=outDir+f"/S{ind}.pth"
            torch.save(Sn, out_S_file)
            print(f"S{ind} saved to {out_S_file}")

        # Step 4: Map the final S_{n+1} to N x N matrix
        final_output = self.final_mapping_layer(Sn)
        final_output = final_output.squeeze(1)  # Remove channel dimension (batch_size, 1, N, N) -> (batch_size, N, N)
        # Step 5: Compute the target scalar E by summing all elements in the N x N matrix
        E = final_output.view(final_output.size(0), -1).sum(dim=1)  # Sum over all elements for each batch

        return E



class toFile_CustomDataset(Dataset):
    def __init__(self, X, Y):
        """
        Custom dataset for supervised learning with `dsnn_qt`.

        Args:
            X (torch.Tensor): Input tensor of shape (num_samples, 3, N, N).
            Y (torch.Tensor): Target tensor of shape (num_samples,).
        """
        self.X = X
        self.Y = Y

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.Y)

    def __getitem__(self, idx):
        """
        Retrieves the input and target at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (input, target) where input is of shape (3, N, N) and target is a scalar.
        """
        return self.X[idx], self.Y[idx]

#test part
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
decrease_over = 50

decrease_rate = 0.9


num_epochs = 1000

decrease_overStr=format_using_decimal(decrease_over)
decrease_rateStr=format_using_decimal(decrease_rate)

in_model_dir=f"./out_model_data/N{N}/C{C}/layer{step_num_after_S1}/"


in_model_file=in_model_dir+f"dsnn_qt_trained_over{decrease_overStr}_rate{decrease_rateStr}_epoch{num_epochs}_num_samples200000.pth"
inDir=f"./train_test_data/N{N}/"

in_pkl_test_file=inDir+"/db.test_num_samples200000.pkl"

with open(in_pkl_test_file,"rb") as fptr:
    X_test, Y_test = pickle.load(fptr)

X_test_array=np.array(X_test)
Y_test_array=np.array(Y_test)
print(f"Y_test_array.shape={Y_test_array.shape}")
batch_size = 1000  # Define batch size
X_test_tensor=torch.tensor(X_test_array,dtype=torch.float)

Y_test_tensor=torch.tensor(Y_test_array,dtype=torch.float)

# Create test dataset and DataLoader
test_dataset = toFile_CustomDataset(X_test_tensor, Y_test_tensor)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)  # Batch size can be adjusted


# Load the trained model
model = toFile_dsnn_qt(
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
model.eval()  # Set model to evaluation mode

which_data=574
single_sample_input = torch.tensor(X_test_array[574], dtype=torch.float).unsqueeze(0).to(device)  # Add batch dimension
single_sample_target = torch.tensor(Y_test_array[574], dtype=torch.float).unsqueeze(0).to(device)  # Add batch dimension

out_S0_file=outDir+"/S0.pth"
torch.save(single_sample_input, out_S0_file)
print(f"S0 saved to {out_S0_file}")

# Initialize S1 for the single sample
S1 = model.initialize_S1(single_sample_input)

# Pass through the model
with torch.no_grad():  # No need for gradients during inference
    prediction = model(single_sample_input, S1, steps=step_num_after_S1)

# Print results
print(f"Input: {single_sample_input}")
print(f"True Target: {single_sample_target.item()}")
print(f"Prediction: {prediction.item()}")