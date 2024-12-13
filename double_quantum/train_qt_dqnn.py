from model_qt_dsnn_config import *


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
    def __init__(self, out_channels=C, kernel_size=filter_size, padding=2):
        super(Phi0Layer, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        self.conv_W0 = nn.Conv2d(
            in_channels=3,
            out_channels=3 * self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            groups=3,
            bias=False  # Bias disabled
        )

        self.conv_W1 = nn.Conv2d(
            in_channels=3,
            out_channels=3 * self.out_channels,  # Same logic as conv_W0
            kernel_size=self.kernel_size,
            padding=self.padding,
            groups=3,  # Ensures separate convolutions for each channel
            bias=False
        )

    def forward(self, x):
        conv_W0 = self.conv_W0(x)
        conv_W1 = self.conv_W1(x)
        batch_size, total_out_channels, H, W = conv_W0.shape
        conv_W0 = conv_W0.view(batch_size, 3, self.out_channels, H, W)
        conv_W1 = conv_W1.view(batch_size, 3, self.out_channels, H, W)
        multiplied = conv_W0 * conv_W1
        Phi0 = multiplied.sum(dim=1)

        return Phi0


set_random_seed(18)

phi0_layer = Phi0Layer(out_channels=C, kernel_size=filter_size, padding=2)

Phi0_output = phi0_layer(X_train_tensor)

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


X_train_transformed = apply_orthogonal_matrix(X_train_tensor, U)

Phi0_output_transformed=phi0_layer(X_train_transformed)
print(Phi0_output_transformed[3])
