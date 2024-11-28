import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
# this script generates data for random energy model
# System Parameters
# System Parameters
L = 50  # Number of spins
r = 5   # Number of spins in each interaction term
K = 60  # Number of interaction terms
N_samples=10000

# seed=17
# Generate random combinations and coefficients
# np.random.seed(seed)  # For reproducibility

# Generate random spin configurations
spin_configurations_samples = np.random.choice([-1, 1], size=(N_samples, L))
# Generate random combinations and interaction coefficients for each spin configuration
combinations_all_samples = [
        [np.random.choice(L, r, replace=False) for _ in range(K)] for _ in range(N_samples)
    ]
J_all_samples = [np.random.normal(0, 1, K) for _ in range(N_samples)]  # Unique J_all for each spin configuration


def compute_energy(spins, K, combinations, J_all):
    """
    Compute the energy for a given spin configuration.

    Parameters:
    spins (array): A 1D array of spin values of length L.
    K (int): Number of interaction terms.
    combinations (list): List of K combinations of r spin indices.
    J_all (array): Array of K interaction coefficients.

    Returns:
    float: The computed energy for the given spin configuration.
    """
    H = 0  # Initialize energy
    for m in range(K):
        # Compute interaction term for combination m
        interaction = np.prod([spins[idx] for idx in combinations[m]])
        H += -J_all[m] * interaction  # Update energy
    return H


def generate_data(combinations_all_samples,spin_configurations_samples,J_all_samples,K, N_samples, train_ratio=0.8):
    """

    :param K (int): Number of interaction terms.
    :param N_samples (int): Total number of samples (training + testing).
    :param train_ratio (float): Proportion of training data.
    :return: tuple: (X_train, Y_train, X_test, Y_test)
    """
    # Compute energies for all configurations
    energies = np.array([
        compute_energy(spins, K, combinations, J_all)
        for spins, combinations, J_all in zip(spin_configurations_samples, combinations_all_samples, J_all_samples)
    ])
    # Split into training and testing datasets
    split_index = int(train_ratio * N_samples)
    X_train, Y_train = spin_configurations_samples[:split_index], energies[:split_index]
    X_test, Y_test = spin_configurations_samples[split_index:], energies[split_index:]
    return X_train, Y_train, X_test, Y_test
tStart=datetime.now()
# Generate training and testing datasets
X_train, Y_train, X_test, Y_test = generate_data(combinations_all_samples,spin_configurations_samples,J_all_samples,K, N_samples)

outDir="./data_rand_energy_spin/"
Path(outDir).mkdir(exist_ok=True,parents=True)
#save training data
fileNameTrain=outDir+"/rand_energy_spin.train.pkl"
with open(fileNameTrain,"wb") as fptr:
    pickle.dump((X_train,Y_train),fptr)


fileNameTest=outDir+"/rand_energy_spin.test.pkl"

with open(fileNameTest, "wb") as f:
    pickle.dump((X_test, Y_test), f)

tEnd=datetime.now()
print("time: ",tEnd-tStart)