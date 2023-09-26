#%%
import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib.image import imread
import figureSupportModule as fsm
import seaborn

np_suffixes = ["01","2", "3", "4", "5", "6", "7", "8", "9", "10"]
arrays = []
for n in np_suffixes:
    array_n = np.loadtxt(f"tmat300_{n}")
    arrays.append(array_n)

mean_array = np.mean(np.stack(arrays), axis=0)
mask = mean_array == 0
ax = seaborn.heatmap(
    mean_array,
    linewidths=0.1,
    # ax=ax,
    #annot=fsm.getCompactedAnnotationsForTmat_percent(data["10"][673]["tmatTD"]),
    annot=True,
    fmt=".3f",
    #fmt="s",
    annot_kws={"fontsize": 8},
    mask=mask,
    # square=True,
    cmap="rocket_r",
    vmax=1,
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
fsm.decorateTmatWithLegend("topDown", range(10), ax=ax, zoom=0.012)#
# %%
import numpy
def calculateTransitions(tmat, time):
    n = tmat.shape[0]
    rates = numpy.zeros_like(tmat)
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
fsm.decorateTmatWithLegend("topDown", range(10), ax=ax, zoom=0.012)#
# %%



# %%
