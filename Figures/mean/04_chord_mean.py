
# %%
# %%
import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib.image import imread
import figureSupportModule as fsm

def onlychoord(
    #axesdict,
    data,
    #T,
    zoom=0.01,
    cbarAx=False,
    #tmatOptions=dict(),
    chordOptions=dict(),
):
    chordOpts = dict(
        colors=fsm.topDownColorMap,
        #ax=axesdict[f"chord{T}"],
        onlyFlux=True,
    )
    chordOpts.update(chordOptions)

    fsm.ChordDiagram(
        data,
        **chordOpts,
    )




np_suffixes = ["01","2", "3", "4", "5", "6", "7", "8", "9", "10"]
trajs = [
    "frame_01-SV_4835-SL_29282-T_300",
    "frame_2-SV_12607-SL_11090-T_300",
    "frame_3-SV_11574-SL_28497-T_300",
    "frame_4-SV_21944-SL_17531-T_300",
    "frame_5-SV_9081-SL_31682-T_300",
    "frame_6-SV_3530-SL_10774-T_300",
    "frame_7-SV_23735-SL_13957-T_300",
    "frame_8-SV_13579-SL_31324-T_300",
    "frame_9-SV_8277-SL_3001-T_300",
    "frame_10-SV_5675-SL_26792-T_300",
]

for np_suffix in np_suffixes:
    NPname = np_suffix
    t = trajs[np_suffixes.index(np_suffix)]
    with h5py.File(f"../../../{NPname}.hdf5", "r") as workFile:
        mask = workFile[f"Trajectories/{t}/Trajectory"][0, :, 2] > 84
    data = {}
    data[NPname] = {}
    classificationFile = f"../../../{NPname}TopBottom.hdf5"
    fsm.loadClassificationTopDown(
        classificationFile, data[NPname], NPname, atomMask=mask
    )
    fsm.loadClassificationTopDown("../../../minimized.hdf5", data[NPname], NPname, atomMask=mask)
    for NP in data:
        for T in data[NP]:
            if T == "Ideal":
                continue
            fsm.addTmatTD(data[NPname][T])
            fsm.addTmatTDNN(data[NPname][T])
    chord673_NP = data[NPname][673]["tmatTDNN"]
    chord300_NP = data[NPname][300]["tmatTDNN"]
    np.savetxt(f"chord673_{NPname}.txt", chord673_NP)
    np.savetxt(f"chord300_{NPname}.txt", chord300_NP)


# Plot per T = 300 e T = 673
np_suffixes = ["01","2", "3", "4", "5", "6", "7", "8", "9", "10"]
arrays300 = []
arrays673 = []
for n in np_suffixes:
    array300_n = np.loadtxt(f"chord300_{n}.txt")
    array673_n = np.loadtxt(f"chord673_{n}.txt")
    arrays300.append(array300_n)
    arrays673.append(array673_n)

mean_array300 = np.mean(np.stack(arrays300), axis=0)
mean_array673 = np.mean(np.stack(arrays673), axis=0)

plt.rcParams.update({"font.size": 45})

# Plot per T = 300
onlychoord(
    mean_array300,
    chordOptions=dict(
        visualizationScale=1.0,
        labels=fsm.topDownLabels,
        labelpos=1.12,
    )
)
plt.savefig("chordmean300.png", bbox_inches="tight", pad_inches=0, dpi=300)

# Plot per T = 673
onlychoord(
    mean_array673,
    chordOptions=dict(
        visualizationScale=1.0,
        labels=fsm.topDownLabels,
        labelpos=1.12,
    )
)
plt.savefig("chordmean673.png", bbox_inches="tight", pad_inches=0, dpi=300)

# %%
