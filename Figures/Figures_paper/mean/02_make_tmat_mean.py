



#%%
import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib.image import imread
import figureSupportModule as fsm
import seaborn

# Suffixes for np data
np_suffixes = ["01", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

# Loop through temperature values (300K and 673K)
for temperature in ["300", "673"]:
    arrays = []

    # Load and process data for each np_suffix
    for n in np_suffixes:
        array_n = np.loadtxt(f"tmat{temperature}_{n}.txt")
        arrays.append(array_n)

    # Calculate the mean array
    mean_array = np.mean(np.stack(arrays), axis=0)

    # Create a mask for zero values
    mask = mean_array == 0

    # Create the heatmap
    ax = seaborn.heatmap(
        mean_array,
        linewidths=0.1,
        annot=True,
        fmt=".1e",  # Scientific notation format
        annot_kws={"fontsize": 6},
        mask=mask,
        cmap="rocket_r",
        vmax=1,
        vmin=0,
        cbar=False,
        square=True,
        xticklabels=fsm.topDownLabels,
        yticklabels=fsm.topDownLabels,
    )

    # Set the axis labels
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)

    # Add a legend
    fsm.decorateTmatWithLegend("topDown", range(10), ax=ax, zoom=0.012)

    # Set the title above the matrix
    plt.title(f"Averaged tmat at {temperature}", fontsize=16, pad=20)

    # Save the heatmap as an image
    plt.savefig(f"averaged_tmat_{temperature}.png", bbox_inches="tight", pad_inches=0, dpi=300)

    # Clear the current plot for the next iteration
    plt.show()
    plt.clf()
    


# %%
def calculateTransitions(tmat, time):
    n = tmat.shape[0]
    rates = np.zeros_like(tmat)
    for i in range(n):
        for j in range(n):
            if i != j and tmat[i, j] != 0:
                rates[i, j] = 1.0 / (tmat[i, j] / time)
    return rates



test673_NP = calculateTransitions(mean_array, 1)
matrix=test673_NP
ax = seaborn.heatmap(
    matrix,
    linewidths=0.1,
    # ax=ax,
    #annot=fsm.getCompactedAnnotationsForTmat_percent(data["10"][673]["tmatTD"]),
    annot=True,
    fmt=".2e",
    #fmt="s",
    annot_kws={"fontsize": 4},
    mask=mask,
    # square=True,
    cmap="rocket_r",
    vmax=100000,
    vmin=0,
    cbar=False,
    square=True,
    xticklabels=fsm.topDownLabels,
    yticklabels=fsm.topDownLabels,
    # annot_kws=dict(
    # weight="bold",
    # ),
)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)