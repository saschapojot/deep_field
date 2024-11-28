import numpy as np
import pickle
from pathlib import Path
import pandas as pd
from scipy.special import binom
from datetime import datetime
#this script computes the mean and variance of random energy model using Jackknife method





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


def generate_data(L,r,K,N_samples):
    """

    :param L: Number of spins
    :param r: Number of spins in each interaction term
    :param K: Number of interaction terms
    :param N_samples: number of samples
    :return:
    """
    # Generate random spin configurations
    spin_configurations_samples = np.random.choice([-1, 1], size=(N_samples, L))

    # Generate random combinations and interaction coefficients for each spin configuration
    combinations_all_samples = [
        [np.random.choice(L, r, replace=False) for _ in range(K)] for _ in range(N_samples)
    ]

    J_all_samples = [np.random.normal(0, 1, K) for _ in range(N_samples)]  # Unique J_all for each spin configuration

    # Compute energies for all configurations
    energies = np.array([
        compute_energy(spins, K, combinations, J_all)
        for spins, combinations, J_all in zip(spin_configurations_samples, combinations_all_samples, J_all_samples)
    ])

    return energies


def jackknife_mean_variance(samples):
    """
    Perform jackknife resampling to estimate the mean and variance.

    Parameters:
        samples (array-like): The dataset from which to calculate jackknife estimates.

    Returns:
        jackknife_mean (float): The jackknife estimate of the mean.
        jackknife_variance (float): The jackknife estimate of the variance of the mean.
    """
    # Number of samples
    N = len(samples)

    # Compute leave-one-out means
    leave_one_out_means = np.zeros(N)
    for i in range(N):
        leave_one_out_means[i] = np.mean(np.delete(samples, i))

    # Compute jackknife mean
    jackknife_mean = np.mean(leave_one_out_means)

    # Compute jackknife variance
    jackknife_variance = (N - 1) / N * np.sum((leave_one_out_means - jackknife_mean) ** 2)

    return jackknife_mean, jackknife_variance




# System Parameters
L = 50  # Number of spins
r = 5   # Number of spins in each interaction term
K = 60  # Number of interaction terms

outPath="./dataJackknife/"
Path(outPath).mkdir(exist_ok=True,parents=True)
sample_num_vec=[10,20,30,40,80,160]+[160*2**n for n in range(1,10)]
print(f"{len(sample_num_vec)} samples")
mean_vec=[]
var_vec=[]
tStart=datetime.now()
counter=0
for n in sample_num_vec:
    energy_vec=generate_data(L,r,K,n)
    mean_tmp,var_tmp=jackknife_mean_variance(energy_vec)
    mean_vec.append(mean_tmp)
    var_vec.append(var_tmp)
    print(f"sample_num_vec[{counter}] processed")
    counter+=1

outData={
    "number_samples":sample_num_vec,
    "mean": mean_vec,
    "var":var_vec
}

df = pd.DataFrame(outData)

df.to_csv(outPath+"/stats.csv",index=False)

tEnd=datetime.now()
print("time: ",tEnd-tStart)