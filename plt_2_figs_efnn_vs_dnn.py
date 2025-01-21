from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

import matplotlib as mpl

from matplotlib.pyplot import yticks

mpl.rcParams['axes.linewidth'] = 2.5  # Set for all plots

# plt.rcParams['font.family'] = 'DejaVu Sans'  # Or any desired font
#this script plots efnn vs dnn, neuron, 2 separate figures


# Enable LaTeX rendering
# mpl.rc('text', usetex=True)
# mpl.rc('font', family='serif', serif=['Computer Modern'])


inCsvDir="./final_sum_more_neurons_inf_range_general_r/compare_layer_neuron_num/"

in_pkl_File="./final_sum_more_neurons_inf_range_general_r/data_inf_range_model_L15_K_455_r3/inf_range.train.pkl"

# in_efnn_vs_dnn_csv_file=inCsvDir+"/efnn_dnn_compare.csv"
layer_num_vec=[1,2,3]
in_more_neurons_csv_file_layer1=inCsvDir+f"/{layer_num_vec[0]}_std_loss.csv"
in_more_neurons_csv_file_layer2=inCsvDir+f"/{layer_num_vec[1]}_std_loss.csv"
in_more_neurons_csv_file_layer3=inCsvDir+f"/{layer_num_vec[2]}_std_loss.csv"


width=6
height=8
textSize=33
yTickSize=33
xTickSize=33
legend_fontsize=23
marker_size1=100
marker_size2=80
lineWidth1=3
lineWidth2=2
tick_length=13
tick_width=5
minor_tick_length=7
minor_tick_width=1
#efnn vs dnn
# fig, ax1 =plt.subplots(1,1)
in_csv_efnn_vs_dnn="./final_sum_inf_range_model_r_general/efnn_vs_dnn_csv/efnn_vs_dnn.csv"
in_df_efnn_vs_dnn=pd.read_csv(in_csv_efnn_vs_dnn)
layerNumVec=np.array(in_df_efnn_vs_dnn["layerNum"])

err_efnn_vec=np.array(in_df_efnn_vs_dnn["efnn"])

err_dnn_vec=np.array(in_df_efnn_vs_dnn["dnn"])
print(f"err_efnn_vec={err_efnn_vec}")
plt.figure(figsize=(width, height))
plt.minorticks_on()
plt.scatter(layerNumVec,err_efnn_vec,color="blue",marker="o",label="EFNN",s=marker_size1)
plt.plot(layerNumVec,err_efnn_vec,color="blue",linestyle="dashed", linewidth=lineWidth1)
plt.yscale("log")

plt.scatter(layerNumVec,err_dnn_vec,color="red",marker="s",label="DNN",s=marker_size1)
plt.plot(layerNumVec,err_dnn_vec,color="red",linestyle="dashed", linewidth=lineWidth1)
custom_yticks = [0.01,0.03,0.07, 1]
plt.yticks(custom_yticks, labels=["0.01", "0.03","0.07", "1"],fontsize=yTickSize)
plt.xticks([1,2,3],labels=["1","2","3"],fontsize=xTickSize)
plt.tick_params(axis='both', length=tick_length,width=2)  # axis='both' adjusts both x and y ticks
plt.tick_params(axis='y', which='minor', length=minor_tick_length, width=minor_tick_width, color='black')   # Minor ticks on y-axis
plt.legend(loc="best",fontsize=legend_fontsize)
plt.gca().yaxis.set_label_position("right")  # Move label to the right
plt.xlabel("Number of FP layers",fontsize=textSize)
plt.ylabel("Relative error",fontsize=textSize)
# plt.title("EFNN vs DNN",fontsize=28)

outDir="./fig_efnn_vs_dnn/"
plt.tight_layout()
plt.savefig(outDir+"/efnn_vs_dnn.svg")
plt.savefig(outDir+"/efnn_vs_dnn.png")
plt.close()


##more neurons
with open(in_pkl_File,"rb") as fptr:
    X_train, Y_train = pickle.load(fptr)
abs_avg_Y_train=np.abs(np.mean(np.array(Y_train)))

in_df_more_neurons_layer1=pd.read_csv(in_more_neurons_csv_file_layer1)
in_df_more_neurons_layer2=pd.read_csv(in_more_neurons_csv_file_layer2)
in_df_more_neurons_layer3=pd.read_csv(in_more_neurons_csv_file_layer3)
# print(in_df_more_neurons)

#layer 1
neuron_num_vec_layer1=in_df_more_neurons_layer1["neuron_num"]
std_loss_vec_layer1=in_df_more_neurons_layer1["std_loss"]
std_loss_vec_layer1=np.array(std_loss_vec_layer1)
relative_error_layer1=std_loss_vec_layer1/abs_avg_Y_train
#layer 2
neuron_num_vec_layer2=in_df_more_neurons_layer2["neuron_num"]
std_loss_vec_layer2=in_df_more_neurons_layer2["std_loss"]
std_loss_vec_layer2=np.array(std_loss_vec_layer2)
relative_error_layer2=std_loss_vec_layer2/abs_avg_Y_train
#layer 3
neuron_num_vec_layer3=in_df_more_neurons_layer3["neuron_num"]
std_loss_vec_layer3=in_df_more_neurons_layer3["std_loss"]
std_loss_vec_layer3=np.array(std_loss_vec_layer3)
relative_error_layer3=std_loss_vec_layer3/abs_avg_Y_train
last_val_layer1=relative_error_layer1[-1]
last_val_layer3=relative_error_layer3[-1]
print(f"relative_error_layer1={last_val_layer1}")
print(f"last_val_layer3={last_val_layer3}")
# fig, ax2 =plt.subplots(1,1)
def more_neuron_model(x,a,b,c):
    return a*x**b+c
#plot layer 1
params_layer1,cov1=curve_fit(more_neuron_model,neuron_num_vec_layer1,relative_error_layer1,p0=[1.0,-1.0,1.0])
a1,b1,c1=params_layer1
print(f"a1={a1},b1={b1},c1={c1}")

plt.figure(figsize=(width, height))
plt.scatter(neuron_num_vec_layer1,relative_error_layer1,color="green",marker="s",s=marker_size2,label="1 FP layer")
#plot original
plt.plot(neuron_num_vec_layer1,relative_error_layer1,color="green",linestyle="dashed",linewidth=lineWidth2)
#plot fit
# plt.plot(neuron_num_vec_layer1,more_neuron_model(neuron_num_vec_layer1,*params_layer1),color="green",linestyle="dashed",linewidth=lineWidth2)


#plot layer 2
plt.scatter(neuron_num_vec_layer2,relative_error_layer2,color="magenta",marker="o",label="2 FP layers",s=marker_size2)
#plot original
plt.plot(neuron_num_vec_layer2,relative_error_layer2,color="magenta",linestyle="dotted",linewidth=lineWidth2)
params_layer2,cov2=curve_fit(more_neuron_model,neuron_num_vec_layer2,relative_error_layer2,p0=[1.0,-1.0,1.0])
# print(f"cov2={cov2}")
a2,b2,c2=params_layer2
print(f"a2={a2},b2={b2},c2={c2}")
#plot fit
# plt.plot(neuron_num_vec_layer2,more_neuron_model(neuron_num_vec_layer2,*params_layer2),color="magenta",linestyle="dotted",linewidth=lineWidth2)
#plot layer 3
plt.scatter(neuron_num_vec_layer3,relative_error_layer3,color="navy",marker="P",label="3 FP layers",s=marker_size2)
#plot original
plt.plot(neuron_num_vec_layer3,relative_error_layer3,color="navy",linestyle="dashdot",linewidth=lineWidth2)
params_layer3,cov3=curve_fit(more_neuron_model,neuron_num_vec_layer3,relative_error_layer3,p0=[1.0,-1.0,1.0])
a3,b3,c3=params_layer3
print(f"a3={a3},b3={b3},c3={c3}")
#plot fit
# plt.plot(neuron_num_vec_layer3,more_neuron_model(neuron_num_vec_layer3,*params_layer3),color="navy",linestyle="dashdot",linewidth=lineWidth2)


plt.yscale("log")
plt.xticks([15,60,105,150],labels=["15","60","105","150"],fontsize=xTickSize)
plt.yticks([0.05,0.02,0.01,0.004],labels=["0.05","0.02","0.01","0.004"],fontsize=yTickSize)
plt.tick_params(axis='y', which='minor', length=minor_tick_length, width=minor_tick_width, color='black')   # Minor ticks on y-axis

plt.xlabel("Neuron number",fontsize=textSize)
plt.ylabel("Relative error",fontsize=textSize)
plt.legend(loc="best",fontsize=legend_fontsize)
# plt.title("3 FP layers, more neurons",fontsize=28)
plt.gca().yaxis.set_label_position("right")  # Move label to the right

plt.tick_params(axis='both', length=tick_length,width=2)  # axis='both' adjusts both x and y ticks
plt.tight_layout()
plt.savefig(outDir+"/more_neurons.svg")
plt.savefig(outDir+"/more_neurons.png")

# plt.savefig(outDir+"/more_neurons_fit.png")
plt.close()