import numpy as np

from config_lin import *
import pickle
import sys
from datetime import datetime
from sklearn.linear_model import LinearRegression
import joblib
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

inDir=f"./train_test_data/N{N}/"

in_pkl_test_file=inDir+"/db.test.pkl"
with open(in_pkl_test_file,"rb") as fptr:
    X_test, Y_test = pickle.load(fptr)

X_test_array = np.array(X_test)  # Shape: (num_samples, 3,N, N, )
Y_test_array = np.array(Y_test)  # Shape: (num_samples,)

lin_model_out_dir="./lin_model_out/"

in_model_fileName=lin_model_out_dir+"/lin.joblib"
with open(in_model_fileName, "rb") as file:
    loaded_model = joblib.load(file)


num_samples = X_test_array.shape[0]
# truncate=20

features_list = []
tFeaturesStart=datetime.now()
for i in range(num_samples):
    Sigma_combined = X_test_array[i]  # Shape: (3, N, N)
    features = generate_interaction_features_combined(
        Sigma_combined, unique_pairs_with_displacements, four_body_combinations_with_displacements
    )
    features_list.append(features)
    if i%1000==0:
        print(f"processed i={i}")


X_test_features=np.array(features_list)
Y_test_final=Y_test_array


tFeaturesEnd=datetime.now()

print(f"feature time: {tFeaturesEnd-tFeaturesStart}")

Y_pred=loaded_model.predict(X_test_features)

# Compare predictions with the true values
r2 = r2_score(Y_test_final, Y_pred)
mse = mean_squared_error(Y_test_final, Y_pred)
mae = mean_absolute_error(Y_test_final, Y_pred)

print(f"R^2 score: {r2}")
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")

# Plot true values vs. predicted values
plt.scatter(Y_test_final, Y_pred, alpha=0.6)
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("True vs. Predicted Values")
plt.plot([Y_test_final.min(), Y_test_final.max()], [Y_test_final.min(), Y_test_final.max()], 'r--')  # Diagonal line
plt.savefig(lin_model_out_dir+"/fit.png")

