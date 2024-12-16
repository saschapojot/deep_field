import itertools
import numpy as np
from scipy.special import binom
# Define the lattice size
N = 10  # Replace with your desired value (must be an even number)

# Generate the 2D indices from 0 to N/2 - 1
half_N = N // 2
print(f"half_N={half_N}")
indices = list(itertools.combinations_with_replacement(range(half_N), 2))
# Generate unique pairs of [i, j] and [k, l] combinations with repetition
unique_pairs = list(itertools.combinations_with_replacement(indices, 2))

# Define displacement vectors for the four tiles
v0 = np.array([0, 0])
v1 = np.array([0, half_N])
v2 = np.array([half_N, 0])
v3 = np.array([half_N, half_N])

displacements = [v0, v1, v2, v3]

# Generate unique pairs with displacements
unique_pairs_with_displacements = []

for displacement in displacements:
    for (i1, j1), (i2, j2) in unique_pairs:
        # Add displacement to both indices
        displaced_pair = ([i1 + displacement[0], j1 + displacement[1]],
                          [i2 + displacement[0], j2 + displacement[1]])
        unique_pairs_with_displacements.append(displaced_pair)



# Generate 4-body combinations from unique_pairs
four_body_combinations = list(itertools.combinations_with_replacement(unique_pairs, 2))
# Generate 4-body combinations with displacements
four_body_combinations_with_displacements = []


for displacement in displacements:
    for ((i1, j1), (i2, j2)), ((k1, l1), (k2, l2)) in four_body_combinations:
        # Apply displacement to each pair
        displaced_combination = (
            ([i1 + displacement[0], j1 + displacement[1]],
             [i2 + displacement[0], j2 + displacement[1]]),
            ([k1 + displacement[0], l1 + displacement[1]],
             [k2 + displacement[0], l2 + displacement[1]])
        )
        four_body_combinations_with_displacements.append(displaced_combination)



def generate_interaction_features_combined(Sigma_combined,unique_pairs_with_displacements,four_body_combinations_with_displacements):
    """
        Generate combined 2-body and 4-body interaction features using precomputed displacements.
        :param Sigma_combined: A 3D array of shape (3, N, N) representing spin components.
        :param unique_pairs_with_displacements: Precomputed unique pairs with displacements for 2-body interactions.
        :param four_body_combinations_with_displacements: Precomputed 4-body combinations with displacements.
        :return: A single feature vector combining 2-body and 4-body features.
        """
    Sigma_x, Sigma_y, Sigma_z = Sigma_combined
    features = []
    # Generate 2-body features
    for (i1, j1), (i2, j2) in unique_pairs_with_displacements:
        # Compute the dot product for the pair
        term = (
                Sigma_x[i1, j1] * Sigma_x[i2, j2]
                + Sigma_y[i1, j1] * Sigma_y[i2, j2]
                + Sigma_z[i1, j1] * Sigma_z[i2, j2]
        )
        features.append(term)

        # Generate 4-body features
    for ((i1, j1), (i2, j2)), ((k1, l1), (k2, l2)) in four_body_combinations_with_displacements:
        # Compute the 4-body term as a product of two dot products
        term = (
                (Sigma_x[i1, j1] * Sigma_x[i2, j2]
                 + Sigma_y[i1, j1] * Sigma_y[i2, j2]
                 + Sigma_z[i1, j1] * Sigma_z[i2, j2])
                *
                (Sigma_x[k1, l1] * Sigma_x[k2, l2]
                 + Sigma_y[k1, l1] * Sigma_y[k2, l2]
                 + Sigma_z[k1, l1] * Sigma_z[k2, l2])
        )
        features.append(term)

    return np.array(features)



