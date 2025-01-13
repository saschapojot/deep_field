import re
import matplotlib.pyplot as plt
import numpy as np
# from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import matplotlib as mpl
from model_qt_dsnn_config import *
# plt.rcParams['font.family'] = 'DejaVu Sans'  # Or any desired font
# mpl.rc('text', usetex=True)
# mpl.rc('font', family='serif', serif=['Computer Modern'])
mpl.rcParams['axes.linewidth'] = 2.5  # Set for all plots
#this script plots the test performance of model trained on 10 by 10 lattice on larger lattices

layer_num=0
C=10
rate=0.9
N_vec=np.array([10,15,20,25,30,35,40])
num_suffix=40000
num_epochs = 1200
inDirRoot="./larger_lattice_test_performance/"

def N_2_test_file(N):
    in_model_dir = f"./larger_lattice_test_performance/N{N}/C{C}/layer{layer_num}/"
    test_txt_file=in_model_dir+f"/custom_err_test_over50_rate{rate}_epoch{num_epochs}_num_samples{num_suffix}.txt"
    return test_txt_file

def N_2_test_data_pkl(N):
    in_model_dir = f"./larger_lattice_test_performance/N{N}/C{C}/layer{layer_num}/"
    pkl_test_file=in_model_dir+f"/db.test_num_samples{num_suffix}.pkl"
    return pkl_test_file

pattern_std=r'std_loss=\s*([+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)'
pattern_custom_err=r'custom_err\s*=\s*([+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)'
def file_2_std(test_fileName):
    with open(test_fileName,"r") as fptr:
        line=fptr.readline()
    # print(line)
    match_std_loss=re.search(pattern_std,line)
    match_custom_err=re.search(pattern_custom_err,line)
    if match_std_loss:
        std_loss= float(match_std_loss.group(1))
    else:
        print("format error")
        exit(12)
    if match_custom_err:
        custom_err=float(match_custom_err.group(1))
    return std_loss,custom_err




file_vec=[N_2_test_file(N) for N in N_vec]
# print(file_vec)

std_loss_vec=[]
custom_err_vec=[]
for file in file_vec:
    std_loss, custom_err=file_2_std(file)
    std_loss_vec.append(std_loss)
    custom_err_vec.append(custom_err)

pkl_file_vec=[N_2_test_data_pkl(N) for N in N_vec]

abs_avg_Y_train_vec=[]
for j in range(0,len(pkl_file_vec)):
    file_pkl_tmp=pkl_file_vec[j]
    with open(file_pkl_tmp,"rb") as fptr:
        X_train_tmp,Y_train_tmp=pickle.load(fptr)
    Y_train_tmp=np.array(Y_train_tmp)
    absTmp=np.abs(np.mean(Y_train_tmp))
    # absTmp=np.std(Y_train_tmp)
    abs_avg_Y_train_vec.append(absTmp)

std_loss_vec=np.array(std_loss_vec)
abs_avg_Y_train=np.array(abs_avg_Y_train_vec)
N_2_vec=np.array(N_vec)**2
relative_error=(std_loss_vec/abs_avg_Y_train)
print(f"std_loss_vec={std_loss_vec}")
print(f"relative error: {relative_error}")
N_vec=np.array(N_vec)
# N_vec=N_vec.reshape(-1, 1)
# log_N = np.log(N_vec).reshape(-1, 1)
log_relative_error = np.log(relative_error)
# Define the model

plt.figure()
plt.scatter(N_vec,relative_error,color="blue",marker="o")
plt.plot(N_vec,relative_error,color="blue",linestyle="dashed")
plt.xlabel("Lattice size")
plt.ylabel("Relative error")
plt.xticks(N_vec)
plt.title(r"Performance of $10\times 10$ model on larger lattice")
plt.savefig(inDirRoot+f"/larger_lattice_test_performance_rate{rate}_layer{layer_num}.png")


plt.close()

custom_err_vec=np.array(custom_err_vec)


def model_custom_err_vs_N(x,a,b,c):
    return a*x**b+c
N_vec_to_fit=N_vec
initial_guess = [1, -1, 1]
params,covariance=curve_fit(model_custom_err_vs_N,N_vec_to_fit,custom_err_vec,initial_guess)
a,b,c=params
print(f"a={a},b={b},c={c}")
# print(f"covariance={covariance}")

plt_fit_data_num=100
plt_custom_err_N_vec=np.linspace(np.min(N_vec),np.max(N_vec),plt_fit_data_num)

plt_custom_err_fit=model_custom_err_vs_N(plt_custom_err_N_vec,a,b,c)

width=6
height=8
textSize=33
yTickSize=33
xTickSize=33
legend_fontsize=33
lineWidth1=3
marker_size1=100
out_db_qt_dir="./larger_lattice_test_performance/"
Path(out_db_qt_dir).mkdir(parents=True, exist_ok=True)
plt.figure(figsize=(width, height))

# Scale the custom error vector for plotting
plt_custom_err_vec = 100 * custom_err_vec

# Plot the data
plt.plot(N_vec, custom_err_vec, color="darkviolet", linestyle="dashed", linewidth=lineWidth1)
plt.scatter(N_vec, custom_err_vec, color="darkviolet", marker="o", label="custom error", s=marker_size1)

print(f"custom_err_vec={custom_err_vec}")

# Axis labels and ticks
plt.ylabel(r"$\epsilon(N)$", fontsize=textSize, labelpad=10)  # Adjust label padding for better spacing
plt.xlabel("$N$", fontsize=textSize)

plt.xticks([10, 20, 30, 40], ["10", "20", "30", "40"], fontsize=xTickSize)
plt.yticks([0.06,0.03, 0.02, 0.01], labels=["0.06","0.03", "0.02", "0.01"], fontsize=yTickSize)

# Adjust layout to fit y-label on the right and remove extra space
plt.tight_layout(rect=[0, 0, 1, 1])  # Prevent truncation and ensure the full figure fits
plt.gca().yaxis.set_label_position("right")  # Move y-axis label to the right

# Save the figure
plt.savefig(out_db_qt_dir + "/custom_error.svg", bbox_inches="tight", dpi=300)  # bbox_inches ensures no truncation
plt.close()
