import numpy as np
import pickle
from scipy.special import binom
import matplotlib.pyplot as plt

#data stats in larger lattice, actually test data
N=10
C=25
step_num_after_S1=0
# inDir=f"./train_test_data/N{N}/"
num_suffix=200000
inDir=f"./train_test_data/N{N}"
in_pkl_train_file=inDir+f"/db.train_num_samples{num_suffix}.pkl"

in_pkl_test_file=inDir+f"/db.test_num_samples{num_suffix}.pkl"

with open(in_pkl_train_file,"rb") as fptr:
    X_train, Y_train=pickle.load(fptr)

with open(in_pkl_test_file,"rb") as fptr:
    X_test, Y_test=pickle.load(fptr)

Y_train_array = np.array(Y_train)  # Shape: (num_samples,)
Y_train_mean = np.mean(Y_train_array)
Y_train_std = np.std(Y_train_array)

Y_test_array = np.array(Y_test)
Y_test_mean = np.mean(Y_test_array)
Y_test_std = np.std(Y_test_array)

outTextFile=inDir+"/stats.txt"
out_content=f"N={N} ,train mean={Y_train_mean}, train std={Y_train_std}\n"\
    +f"test mean={Y_test_mean}, test std={Y_test_std}\n"
with open(outTextFile,"w") as fptr:
    fptr.write(out_content)

# Create the histogram
plt.hist(Y_train_array, bins=30, edgecolor='black')  # bins: number of bins; edgecolor: outline color
plt.xlabel('Value')  # Label for the x-axis
plt.ylabel('#')  # Label for the y-axis
plt.title(f'Histogram of E, N={N}, mean={Y_train_mean}, std={Y_train_std}')  # Title of the plot
plt.savefig(inDir+f"/y_train_stats_N{N}.png")
plt.close()

plt.hist(Y_test_array, bins=30, edgecolor='black')  # bins: number of bins; edgecolor: outline color
plt.xlabel('Value')  # Label for the x-axis
plt.ylabel('#')  # Label for the y-axis
plt.title(f'Histogram of E, N={N}, mean={Y_test_mean}, std={Y_test_std}')  # Title of the plot
plt.savefig(inDir+f"/y_test_stats_N{N}.png")
plt.close()

# plt.hist((Y_train_array-Y_train_mean)/Y_train_std, bins=30, edgecolor='black')
# plt.xlabel('Value')  # Label for the x-axis
# plt.ylabel('#')  # Label for the y-axis
# plt.title(f'Histogram of E, N={N}')
# plt.savefig(inDir+f"/y_train_stats_N{N}_normalize.png")
# plt.close()
