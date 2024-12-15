import numpy as np
import pickle
from scipy.special import binom
import matplotlib.pyplot as plt


N=10


inDir=f"./train_test_data/N{N}/"

in_pkl_train_file=inDir+"/db.train.pkl"

with open(in_pkl_train_file,"rb") as fptr:
    X_train, Y_train=pickle.load(fptr)

Y_train_array = np.array(Y_train)  # Shape: (num_samples,)
# Create the histogram
plt.hist(Y_train_array, bins=30, edgecolor='black')  # bins: number of bins; edgecolor: outline color
plt.xlabel('Value')  # Label for the x-axis
plt.ylabel('#')  # Label for the y-axis
plt.title(f'Histogram of E, N={N}')  # Title of the plot
plt.savefig(f"y_train_stats_N{N}.png")