import matplotlib.pyplot as plt
import pickle
import numpy as np

import pandas as pd


inPath="./dataJackknife/"


inCsvFile=inPath+"/stats.csv"

# Load the CSV file
df_loaded = pd.read_csv(inCsvFile)

# Display the DataFrame

number_samples_vec=np.array(df_loaded["number_samples"])

mean_vec=np.array(df_loaded["mean"])

var_vec=np.array(df_loaded["var"])

plt.figure()
plt.scatter(number_samples_vec,mean_vec,color="blue",label="mean")
plt.title("Mean vs number of samples")
plt.xlabel("Number of samples")
plt.xscale("log")
plt.ylabel("Mean")
plt.legend(loc="best")
plt.savefig(inPath+"/mean.png")
plt.close()


plt.figure()
plt.scatter(number_samples_vec,var_vec,color="green",label="var")
plt.title("Var vs number of samples")
plt.xlabel("Number of samples")
plt.ylabel("Var")
plt.xscale("log")
plt.legend(loc="best")
plt.savefig(inPath+"/var.png")
plt.close()