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


class Phi0Layer(nn.Module):
    def __init__(self, out_channels, kernel_size, padding=2):
        """
        A modified Phi0Layer where the convolutional matrices (weights) are shared
        across the three input channels (Sigma_x, Sigma_y, Sigma_z).

        Args:
            out_channels (int): Number of output channels per input channel.
            kernel_size (int): Size of the convolution kernel.
            padding (int): Padding for the convolution operation. Default is 2.
        """
        super(Phi0Layer, self).__init__()

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

    def forward(self, x):
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

        return Phi0


class TLayer(nn.Module):
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
        super(TLayer, self).__init__()

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

    def forward(self, x):
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

        return T_result


class NonlinearLayer(nn.Module):
    def __init__(self, in_channels, conv1_out_channels, conv2_out_channels, kernel_size=3, padding=1):
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
        super(NonlinearLayer, self).__init__()

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

    def forward(self, x):
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

        return x


class RecursiveNeuralNetwork(nn.Module):
    def __init__(self, input_channels,
                 phi0_out_channels=C, T_out_channels=C,
                 nonlinear_conv1_out_channels=C, nonlinear_conv2_out_channels=C,
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
        super(RecursiveNeuralNetwork, self).__init__()
        # TLayer
        self.T_layer = TLayer(out_channels=T_out_channels, kernel_size=filter_size)

        # NonlinearLayer for S_n and F_n+1
        self.nonlinear_layer = NonlinearLayer(
            in_channels=nonlinear_conv2_out_channels,
            conv1_out_channels=nonlinear_conv1_out_channels,
            conv2_out_channels=nonlinear_conv2_out_channels,
            kernel_size=3,  # Default convolution kernel size
            padding=1
        )

        # NonlinearLayer for processing input after TLayer
        self.nonlinear_layer_input = NonlinearLayer(
            in_channels=T_out_channels,
            conv1_out_channels=nonlinear_conv1_out_channels,
            conv2_out_channels=nonlinear_conv2_out_channels,
            kernel_size=3,
            padding=1
        )

        # Final mapping layer to N x N matrix
        self.final_mapping_layer = NonlinearLayer(
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
        F1 = self.nonlinear_layer(phi0_output)

        # Step 2: Pass input through TLayer and NonlinearLayer
        T_output = self.T_layer(x)
        nonlinear_output = self.nonlinear_layer_input(T_output)

        # Step 3: Compute S1 as pointwise multiplication of F1 and nonlinear_output
        S1 = F1 * nonlinear_output

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
        for _ in range(steps):
            # Step 1: Compute F_{n+1} by passing S_n through NonlinearLayer
            Fn_plus_1 = self.nonlinear_layer(Sn)

            # Step 2: Pass input through TLayer and NonlinearLayer
            T_output = self.T_layer(x)
            nonlinear_output = self.nonlinear_layer_input(T_output)

            # Step 3: Compute S_{n+1} as pointwise multiplication of Fn_plus_1 and nonlinear_output
            Sn = Fn_plus_1 * nonlinear_output

        # Step 4: Map the final S_{n+1} to N x N matrix
        final_output = self.final_mapping_layer(Sn)
        final_output = final_output.squeeze(1)  # Remove channel dimension (batch_size, 1, N, N) -> (batch_size, N, N)
        # Step 5: Compute the target scalar E by summing all elements in the N x N matrix
        E = final_output.view(final_output.size(0), -1).sum(dim=1)  # Sum over all elements for each batch

        return E

set_random_seed(18)

# phi0_layer = Phi0Layer(out_channels=C, kernel_size=filter_size, padding=2)
# T_layer=TLayer(out_channels=C,kernel_size=filter_size,padding=2)
# Phi0_output = phi0_layer(X_train_tensor)
# T_out=T_layer(X_train_tensor)
# print(T_out[3])
# print(Phi0_output[3])
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


# X_train_transformed = apply_orthogonal_matrix(X_train_tensor, U)
#
# Phi0_output_transformed=phi0_layer(X_train_transformed)
# print(Phi0_output_transformed[3])

# T_out_transformed=T_layer(X_train_transformed)
# print(T_out_transformed[3])


# Instantiate the network
model = RecursiveNeuralNetwork(
    input_channels=3,
    phi0_out_channels=C,
    T_out_channels=C,
    nonlinear_conv1_out_channels=C,
    nonlinear_conv2_out_channels=C,
    final_out_channels=1,
    filter_size=filter_size
)


inDir=f"./train_test_data/N{N}/C{C}"

in_pkl_train_file=inDir+"/db.train.pkl"

with open(in_pkl_train_file,"rb") as fptr:
    X_train, Y_train=pickle.load(fptr)


# Convert to NumPy arrays

X_train_array = np.array(X_train)  # Shape: (num_samples, 3,N, N, )
Y_train_array = np.array(Y_train)  # Shape: (num_samples,)

# Convert NumPy arrays to PyTorch tensors with dtype=torch.float64
X_train_tensor = torch.tensor(X_train_array, dtype=torch.float)  # Shape: (num_samples, 3, N, N)
Y_train_tensor = torch.tensor(Y_train_array, dtype=torch.float)  # Shape: (num_samples,)

batch_size=len(Y_train_array)

# Initialize S1
S1 = model.initialize_S1(X_train_tensor)
