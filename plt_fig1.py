from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm

from matplotlib.colors import TwoSlopeNorm

np.random.seed(190)  # Replace 42 with any integer
# Step 1: Create a 10x10 random matrix with values 1 or -1

N=10
matrix = np.random.choice([1, -1], size=(N, N))

# Set J to 1
J = 1





# Define a colormap with discrete black and white colors
cmap = ListedColormap(['black', 'white'])
bounds = [-1.5, 0, 1.5]  # Boundaries for discrete colors
norm = BoundaryNorm(bounds, cmap.N)

# Create the plot
fig, ax = plt.subplots(figsize=(6, 6))
img = ax.imshow(matrix, cmap=cmap, norm=norm, interpolation='nearest')

# Remove ticks and labels from the main plot
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('')
ax.set_ylabel('')

# Add the colorbar with specified ticks and labels
cbar = fig.colorbar(
    img,
    ax=ax,
    orientation='vertical',
    ticks=[-1, 1],
    fraction=0.046,  # Default is 0.046; adjust as needed
    pad=0.04         # Default is 0.04; reduce to bring colorbar closer
)

# Set the tick labels for the colorbar
cbar.ax.set_yticklabels(['-1', '1'])

# Remove tick lines but retain labels on the colorbar
cbar.ax.tick_params(length=0, which='both', labelsize=12)  # Adjust labelsize as needed

# Optionally, position the labels on one side only (e.g., left)
cbar.ax.yaxis.set_ticks_position('right')  # Options: 'left', 'right', 'both', 'none'
cbar.ax.yaxis.set_label_position('right')

outDir="./fig1/"
Path(outDir).mkdir(exist_ok=True,parents=True)
plt.savefig(outDir+"spin_lattice.pdf")
plt.close()

effective_field=np.zeros((N,N))

for i in range(0,N):
    for j in range(0,N):
        i_plus_1=(i+1)%N
        i_minus_1=(i-1)%N
        j_plus_1=(j+1)%N
        j_minus_1=(j-1)%N

        s0=matrix[i_plus_1,j]

        s1=matrix[i_minus_1,j]

        s2=matrix[i,j_plus_1]

        s3=matrix[i,j_minus_1]

        effective_field[i,j]=-J/2*(s0+s1+s2+s3)


# Define discrete colormap for effective_field
# Colors: Blue (-2), Light Blue (-1), White (0), Light Coral (1), Red (2)
colors = ['blue', 'lightblue', 'white', 'lightcoral', 'red']
cmap = ListedColormap(colors)

# Define boundaries and normalization
# Each interval corresponds to one discrete color
bounds = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
norm = BoundaryNorm(bounds, cmap.N)

# Create the plot
fig, ax = plt.subplots(figsize=(6, 6))

# Plot the effective_field using imshow with the discrete colormap
img = ax.imshow(effective_field, cmap=cmap, norm=norm, interpolation='nearest')

ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('')
ax.set_ylabel('')

# Add the colorbar with discrete colors
cbar = fig.colorbar(
    img,
    ax=ax,
    orientation='vertical',
    ticks=[-2, -1, 0, 1, 2],
    fraction=0.046,  # Adjusts the size of the colorbar relative to the plot
    pad=0.04         # Adjusts the space between the plot and the colorbar
)

# Set the tick labels for the colorbar
cbar.ax.set_yticklabels(['-2', '-1', '0', '1', '2'], fontsize=12)

# Remove tick lines but retain labels on the colorbar
cbar.ax.tick_params(length=0, which='both')  # Removes the tick markers

# Position the colorbar labels on one side only
cbar.ax.yaxis.set_ticks_position('right')     # Positions ticks on the right
cbar.ax.yaxis.set_label_position('right')    # Positions labels on the right

# Add a black edge (spine) around the colorbar
for spine in cbar.ax.spines.values():
    spine.set_visible(True)                 # Make spines visible
    spine.set_edgecolor('black')            # Set spine color to black
    spine.set_linewidth(1)                   # Set spine width



plt.savefig(outDir+"effective_field.pdf")
plt.close()

quasi_spin=np.zeros((N,N))

for i in range(0,N):
    for j in range(0,N):
        quasi_spin[i,j]=matrix[i,j]*effective_field[i,j]

# Create the plot
fig, ax = plt.subplots(figsize=(6, 6))
img = ax.imshow(quasi_spin, cmap=cmap, norm=norm, interpolation='nearest')
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('')
ax.set_ylabel('')
# Add the colorbar with discrete colors
cbar = fig.colorbar(
    img,
    ax=ax,
    orientation='vertical',
    ticks=[-2, -1, 0, 1, 2],
    fraction=0.046,  # Adjusts the size of the colorbar relative to the plot
    pad=0.04         # Adjusts the space between the plot and the colorbar
)
# Set the tick labels for the colorbar
cbar.ax.set_yticklabels(['-2', '-1', '0', '1', '2'], fontsize=12)

# Remove tick lines but retain labels on the colorbar
cbar.ax.tick_params(length=0, which='both')  # Removes the tick markers

# Position the colorbar labels on one side only
cbar.ax.yaxis.set_ticks_position('right')     # Positions ticks on the right
cbar.ax.yaxis.set_label_position('right')    # Positions labels on the right

# Add a black edge (spine) around the colorbar
for spine in cbar.ax.spines.values():
    spine.set_visible(True)                 # Make spines visible
    spine.set_edgecolor('black')            # Set spine color to black
    spine.set_linewidth(1)                   # Set spine width

plt.savefig(outDir+"quasi_spin.pdf")
plt.close()
