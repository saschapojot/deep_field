from config_lin import *
import pickle
import sys
from datetime import datetime
from sklearn.linear_model import LinearRegression
import joblib
from pathlib import Path

inDir=f"./train_test_data/N{N}/"

in_pkl_train_file=inDir+"/db.train.pkl"

with open(in_pkl_train_file,"rb") as fptr:
    X_train, Y_train=pickle.load(fptr)


X_train_array = np.array(X_train)  # Shape: (num_samples, 3,N, N, )
Y_train_array = np.array(Y_train)  # Shape: (num_samples,)

tFeaturesStart=datetime.now()
# Extract features for all samples in X_train_array
num_samples = X_train_array.shape[0]
# truncate=100
features_list = []

for i in range(num_samples):
    Sigma_combined = X_train_array[i]  # Shape: (3, N, N)
    features = generate_interaction_features_combined(
        Sigma_combined, unique_pairs_with_displacements, four_body_combinations_with_displacements
    )
    features_list.append(features)
    if i%1000==0:
        print(f"processed i={i}")


# Convert features list to a NumPy array
X_train_features = np.array(features_list)  # Shape: (num_samples, num_features)

tFeaturesEnd=datetime.now()

print(f"feature time: {tFeaturesEnd-tFeaturesStart}")

lin_model_out_dir="./lin_model_out/"
Path(lin_model_out_dir).mkdir(exist_ok=True,parents=True)
features_out_file = lin_model_out_dir + "/X_train_features.pkl"
with open(features_out_file, "wb") as fptr:
    pickle.dump(X_train_features, fptr)
print(f"X_train_features saved to {features_out_file}")