import numpy as np
import pickle
from scipy.special import binom
import matplotlib.pyplot as plt

inPath="./data_rand_energy_spin_T_1.5/"


in_pkl_File=inPath+"/rand_energy_spin.train.pkl"

with open(in_pkl_File,"rb") as fptr:
    x_train,y_train=pickle.load(fptr)

# Create the histogram
plt.hist(y_train, bins=30, edgecolor='black')  # bins: number of bins; edgecolor: outline color
plt.xlabel('Value')  # Label for the x-axis
plt.ylabel('#')  # Label for the y-axis
plt.title('Histogram of y_train')  # Title of the plot
plt.savefig("y_train_stats.png")
