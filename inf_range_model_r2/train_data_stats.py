import numpy as np
import pickle
from scipy.special import binom
import matplotlib.pyplot as plt

# System Parameters
L = 15  # Number of spins
r = 2   # Number of spins in each interaction term

data_inDir=f"./data_inf_range_model_L{L}_r{r}/"
fileNameTrain=data_inDir+"/inf_range.train.pkl"

with open(fileNameTrain,"rb") as fptr:
    x_train,y_train=pickle.load(fptr)

# Create the histogram
plt.hist(y_train, bins=30, edgecolor='black')  # bins: number of bins; edgecolor: outline color
plt.xlabel('Value')  # Label for the x-axis
plt.ylabel('#')  # Label for the y-axis
plt.title('Histogram of y_train')  # Title of the plot
plt.savefig("y_train_stats.png")
