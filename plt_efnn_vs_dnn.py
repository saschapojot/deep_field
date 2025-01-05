import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

import matplotlib as mpl

# Enable LaTeX rendering
mpl.rc('text', usetex=True)
mpl.rc('font', family='serif', serif=['Computer Modern'])
#this script plots efnn vs dnn, more neurons for efnn

inCsvDir="./more_neurons_inf_range_general_r/compare_neuron_num/"

in_pkl_File="./more_neurons_inf_range_general_r/data_inf_range_model_L15_K_455_r3/inf_range.train.pkl"

in_efnn_vs_dnn_csv_file=inCsvDir+"/efnn_dnn_compare.csv"

in_more_neurons_csv_file=inCsvDir+"/std_loss.csv"


in_df_efnn_vs_dnn=pd.read_csv(in_efnn_vs_dnn_csv_file)

layerNumVec=in_df_efnn_vs_dnn["layerNum"]

err_efnn_vec=in_df_efnn_vs_dnn["efnn"]

err_dnn_vec=in_df_efnn_vs_dnn["dnn"]
###efnn vs dnn
fig = plt.figure(figsize=(10, 5))  # Adjust the size as needed
ax1 = fig.add_subplot(1, 2, 1)
ax1.scatter(layerNumVec,err_efnn_vec,color="blue",marker="o",label="EFNN")
ax1.plot(layerNumVec,err_efnn_vec,color="blue",linestyle="dashed")
ax1.set_yscale("log")

ax1.scatter(layerNumVec,err_dnn_vec,color="red",marker="s",label="DNN")
ax1.plot(layerNumVec,err_dnn_vec,color="red",linestyle="dashed")
custom_yticks = [0.06, 0.1, 1]
ax1.set_yticks(custom_yticks, labels=["0.06", "0.1", "1"])
ax1.set_xticks([1,2,3])
ax1.legend(loc="best")
ax1.set_xlabel("Number of FP layers")
ax1.set_ylabel("Relative error",fontsize=14)
ax1.set_title("EFNN vs DNN")
ax1.text(-0.2, 1.1, r"\bfseries (a)",  transform=ax1.transAxes,
         size=12, weight='extra bold', va='top', ha='left')


##more neurons
with open(in_pkl_File,"rb") as fptr:
    X_train, Y_train = pickle.load(fptr)
abs_avg_Y_train=np.abs(np.mean(np.array(Y_train)))

in_df_more_neurons=pd.read_csv(in_more_neurons_csv_file)
# print(in_df_more_neurons)

neuron_num_vec=in_df_more_neurons["neuron_num"]
std_loss_vec=in_df_more_neurons["std_loss"]
std_loss_vec=np.array(std_loss_vec)
relative_error=std_loss_vec/abs_avg_Y_train

ax2 = fig.add_subplot(1, 2, 2)
ax2.scatter(neuron_num_vec,relative_error,color="green",marker="s")
ax2.plot(neuron_num_vec,relative_error,color="green",linestyle="dashed")
ax2.set_xticks([15,50,100,150,200])
ax2.set_yticks([0.07,0.05,0.03])
ax2.set_xlabel("Neuron number")
ax2.set_ylabel("Relative error",fontsize=14)
ax2.set_title("3 FP layers, more neurons")
ax2.yaxis.set_label_position("right")  # Move y-label to the right
ax2.text(1.05, 1.1, r"\bfseries (b)", transform=ax1.transAxes,
         size=12, weight='extra bold', va='top', ha='left')
out_pic_dir="./fig_efnn_vs_dnn/"
Path(out_pic_dir).mkdir(parents=True, exist_ok=True)
plt.savefig(out_pic_dir+"/efnn_vs_dnn.pdf")
plt.close()


