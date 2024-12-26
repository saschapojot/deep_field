import numpy as np
import re
import matplotlib.pyplot as plt
from more_neurons_model_dsnn_config import  decrease_rate,batch_size,learning_rate,weight_decay
from more_neurons_model_dsnn_config import L,r,decrease_over,K
import pickle
from pathlib import Path
import pandas as pd

#this script plots std_loss/abs(avg)

inPath="./compare_neuron_num/"

inCsvName=inPath+"/std_loss.csv"

inDataPath=f"./data_inf_range_model_L{L}_K_{K}_r{r}/"

in_train_pkl_file=inDataPath+"/inf_range.train.pkl"


with open(in_train_pkl_file, 'rb') as fptr:
    X_train, Y_train = pickle.load(fptr)


abs_avg_Y_train=np.abs(np.mean(np.array(Y_train)))


in_df=pd.read_csv(inCsvName)


neuron_num_vec=np.array(in_df["neuron_num"])
std_loss_vec=np.array(in_df["std_loss"])

relative_error=std_loss_vec/abs_avg_Y_train

plt.figure()
plt.scatter(neuron_num_vec,relative_error,color="green",marker="s")
plt.plot(neuron_num_vec,relative_error,color="green",linestyle="dashed")

plt.xlabel("Neuron number")
plt.ylabel("Relative error")
plt.title("Performance on test set")
plt.gca().yaxis.set_label_position("right")  # Move label to the right
plt.savefig(inPath+"/neuron_compare.png")