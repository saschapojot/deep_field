import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import FancyArrowPatch
from matplotlib import gridspec
from pathlib import Path


def plot_spin_lattice(ax, N=10, seed=190, J=1):
    np.random.seed(seed)
    matrix = np.random.choice([1, -1], size=(N, N))

    cmap = ListedColormap(['black', 'white'])
    bounds = [-1.5, 0, 1.5]
    norm = BoundaryNorm(bounds, cmap.N)

    img = ax.imshow(matrix, cmap=cmap, norm=norm, interpolation='nearest')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('Interacting spins', fontsize=14)

    # Add colorbar
    cbar = plt.colorbar(img, ax=ax, orientation='vertical', ticks=[-1, 1],
                        fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(['-1', '1'], fontsize=10)
    cbar.ax.tick_params(length=0, which='both', labelsize=10)
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('right')

    return matrix


def plot_effective_field(ax, matrix, J=1, N=10):
    effective_field = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            i_plus_1 = (i + 1) % N
            i_minus_1 = (i - 1) % N
            j_plus_1 = (j + 1) % N
            j_minus_1 = (j - 1) % N

            s0 = matrix[i_plus_1, j]
            s1 = matrix[i_minus_1, j]
            s2 = matrix[i, j_plus_1]
            s3 = matrix[i, j_minus_1]

            effective_field[i, j] = -J / 2 * (s0 + s1 + s2 + s3)

    # Define discrete colormap
    colors = ['blue', 'lightblue', 'white', 'lightcoral', 'red']
    cmap = ListedColormap(colors)
    bounds = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    norm = BoundaryNorm(bounds, cmap.N)

    img = ax.imshow(effective_field, cmap=cmap, norm=norm, interpolation='nearest')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('Effective Field', fontsize=14)

    # Add colorbar
    cbar = plt.colorbar(img, ax=ax, orientation='vertical', ticks=[-2, -1, 0, 1, 2],
                        fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(['-2', '-1', '0', '1', '2'], fontsize=10)
    cbar.ax.tick_params(length=0, which='both')
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('right')

    # Add black edge around colorbar
    for spine in cbar.ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(1)

    return effective_field


def plot_quasi_spin(ax, matrix, effective_field, J=1, N=10):
    quasi_spin = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            quasi_spin[i, j] = matrix[i, j] * effective_field[i, j]

    # Reuse the colormap and normalization from effective_field
    colors = ['blue', 'lightblue', 'white', 'lightcoral', 'red']
    cmap = ListedColormap(colors)
    bounds = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    norm = BoundaryNorm(bounds, cmap.N)

    img = ax.imshow(quasi_spin, cmap=cmap, norm=norm, interpolation='nearest')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('Independent spins', fontsize=14)

    # Add colorbar
    cbar = plt.colorbar(img, ax=ax, orientation='vertical', ticks=[-2, -1, 0, 1, 2],
                        fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(['-2', '-1', '0', '1', '2'], fontsize=10)
    cbar.ax.tick_params(length=0, which='both')
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('right')

    # Add black edge around colorbar
    for spine in cbar.ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(1)

    return quasi_spin

def create_combined_figure():
    # Create the main figure
    fig = plt.figure(figsize=(14, 12))  # Adjust figure size as needed
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 0.5], hspace=0.1, wspace=0.3)

    # Add subplots
    ax_matrix = fig.add_subplot(gs[0, 0])  # Top-left for Matrix figure
    ax_effective = fig.add_subplot(gs[0, 1])  # Top-right for Effective Field figure
    ax_quasi = fig.add_subplot(gs[1, 1])  # Bottom-center for Quasi Spin figure

    # Plot the first figure: Spin Lattice
    matrix = plot_spin_lattice(ax_matrix)

    # Plot the second figure: Effective Field
    effective_field = plot_effective_field(ax_effective, matrix)

    # Plot the third figure: Quasi Spin
    plot_quasi_spin(ax_quasi, matrix, effective_field)

    # Draw an arrow from Matrix Figure to Effective Field Figure
    start_pos_matrix = (1.15, 0.5)  # Slightly left of right edge, center vertically
    end_pos_effective = (-0.05, 0.5)  # Slightly right of left edge, center vertically

    # Convert to figure-relative coordinates
    start_xy_matrix = ax_matrix.transAxes.transform(start_pos_matrix)
    end_xy_effective = ax_effective.transAxes.transform(end_pos_effective)

    # Convert coordinates to figure coordinates
    start_xy_matrix_fig = fig.transFigure.inverted().transform(start_xy_matrix)
    end_xy_effective_fig = fig.transFigure.inverted().transform(end_xy_effective)

    # Draw the arrow
    arrow1 = FancyArrowPatch(
        start_xy_matrix_fig,
        end_xy_effective_fig,
        transform=fig.transFigure,
        arrowstyle='-|>',  # Triangle arrowhead style
        mutation_scale=20,  # Adjust size of the arrowhead
        color='black'
    )
    fig.patches.append(arrow1)

    # Draw an arrow from Effective Field to Quasi Spin Figure (upward to downward)
    start_pos_effective = (0.5, 0.1)  # Bottom-center of Effective Field Figure
    end_pos_quasi = (0.5, 0.95)  # Top-center of Quasi Spin Figure

    # Convert to figure-relative coordinates
    start_xy_effective = ax_effective.transAxes.transform(start_pos_effective)
    end_xy_quasi = ax_quasi.transAxes.transform(end_pos_quasi)

    # Convert coordinates to figure coordinates
    start_xy_effective_fig = fig.transFigure.inverted().transform(start_xy_effective)
    end_xy_quasi_fig = fig.transFigure.inverted().transform(end_xy_quasi)

    # Draw the downward arrow
    arrow2 = FancyArrowPatch(
        start_xy_effective_fig,
        end_xy_quasi_fig,
        transform=fig.transFigure,
        arrowstyle='-|>',  # Triangle arrowhead style
        mutation_scale=20,  # Adjust size of the arrowhead
        color='black'
    )
    fig.patches.append(arrow2)

    # Add a custom "Energy" box to the right of Quasi Spin
    energy_x = 0.75  # Adjust to position box horizontally
    energy_y = 0.2  # Adjust to position box vertically (centered at half the height of Quasi Spin)
    energy_width = 0.08  # Width of the box
    energy_height = 0.1  # Height of the box

    # Create the rectangle representing the "Energy" box
    rect = plt.Rectangle((energy_x, energy_y), energy_width, energy_height, transform=fig.transFigure,
                         edgecolor='black', facecolor='lightgrey', lw=2)
    fig.patches.append(rect)

    # Add the label "Energy" inside the box
    fig.text(energy_x + energy_width / 2, energy_y + energy_height / 2, 'Energy',
             ha='center', va='center', fontsize=14, fontweight='bold')

    # Add a title for the entire figure
    fig.suptitle('Matrix, Effective Field, Quasi Spin, and Energy', fontsize=16, fontweight='bold')

    # Save the combined figure
    outDir = "./fig1/"
    Path(outDir).mkdir(exist_ok=True, parents=True)
    plt.savefig(outDir + "combined_with_energy_and_arrows_debugged.pdf", bbox_inches='tight')
    # plt.show()




create_combined_figure()