import re
import matplotlib.pyplot as plt
from model_qt_dsnn_config import *

#this script compares test loss for the same N, same layer number, different C values


N=10
#layer-1
step_num_after_S1=5
decrease_over = 50
decrease_rate = 0.6
num_epochs = 1000

C_vec=[10,20,30,40]


decrease_overStr=format_using_decimal(decrease_over)
decrease_rateStr=format_using_decimal(decrease_rate)
suffix_str=f"_over{decrease_overStr}_rate{decrease_rateStr}_epoch{num_epochs}"


def C_2_test_file(C):
    in_model_dir = f"./out_model_data/N{N}/C{C}/layer{step_num_after_S1}/"
    test_fileName = in_model_dir + f"/test{suffix_str}.txt"
    return test_fileName

pattern_std=r'std_loss=\s*([+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)'
def file_2_std(test_fileName):
    with open(test_fileName,"r") as fptr:
        line=fptr.readline()

    match_std_loss=re.search(pattern_std,line)
    if match_std_loss:
        return float(match_std_loss.group(1))
    else:
        print("format error")
        exit(12)




file_vec=[C_2_test_file(C) for C in C_vec]

std_vec=[file_2_std(test_fileName) for test_fileName  in file_vec]

inDir=f"./train_test_data/N{N}/"
in_pkl_train_file=inDir+"/db.train.pkl"
with open(in_pkl_train_file,"rb") as fptr:
    X_train, Y_train=pickle.load(fptr)

Y_train_array = np.array(Y_train)  # Shape: (num_samples,)

Y_train_avg=np.mean(Y_train_array)

abs_Y_train_avg=np.abs(Y_train_avg)

relative_acc=np.array(std_vec)/abs_Y_train_avg

out_pic_dir=f"./compare/N{N}/layer{step_num_after_S1}/"
Path(out_pic_dir).mkdir(exist_ok=True,parents=True)

outFileName=out_pic_dir+f"/pic{suffix_str}.png"

plt.figure()
plt.scatter(C_vec,relative_acc,color="black",s=10)
# Line connecting the points
plt.title(f"Testing set relative error, N={5}, n={step_num_after_S1+1}")
plt.plot(C_vec, relative_acc, linestyle='--', color='black', label='Line',linewidth=0.7)
plt.xlabel("C value")
plt.ylabel("Relative error")
plt.savefig(outFileName)
plt.close()