import numpy as np
import pickle
from scipy.special import binom
import matplotlib.pyplot as plt
from model_dsnn_config import *
# System Parameters
# L = 15  # Number of spins
# K=40
# r = 5   # Number of spins in each interaction term

data_inDir=f"./data_inf_range_model_L{L}_K_{K}_r{r}/"
fileNameTrain=data_inDir+"/inf_range.train.pkl"

with open(fileNameTrain,"rb") as fptr:
    x_train,y_train=pickle.load(fptr)

print("x_train.shape="+str(x_train.shape))
# Create the histogram
plt.hist(y_train, bins=30, edgecolor='black')  # bins: number of bins; edgecolor: outline color
plt.xlabel('Value')  # Label for the x-axis
plt.ylabel('#')  # Label for the y-axis
plt.title('Histogram of y_train')  # Title of the plot
plt.savefig("y_train_stats.png")
plt.close()

fileNameTest=data_inDir+"/inf_range.test.pkl"

with open(fileNameTest,"rb") as fptr:
    x_test, y_test = pickle.load(fptr)


# Create the histogram
plt.hist(y_test, bins=30, edgecolor='black')  # bins: number of bins; edgecolor: outline color
plt.xlabel('Value')  # Label for the x-axis
plt.ylabel('#')  # Label for the y-axis
plt.title('Histogram of y_test')  # Title of the plot
plt.savefig("y_test_stats.png")
plt.close()
y_train_mean=np.mean(y_train)

print(f"y_train_mean={y_train_mean}")