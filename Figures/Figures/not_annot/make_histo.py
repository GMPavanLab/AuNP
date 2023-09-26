
#%%
import matplotlib.pyplot as plt
import h5py
from matplotlib.image import imread
import figureSupportModule as fsm

#%%
np = ["01","2", "3", "4", "5", "6", "7", "8", "9", "10"]
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


for (NPname, t) in zip(np, trajs):
    with h5py.File(f"../../../{NPname}.hdf5", "r") as workFile:
        mask = workFile[f"Trajectories/{t}/Trajectory"][0, :, 2] > 84
    data = {}
    data[NPname] = dict()
    classificationFile = f"../../../{NPname}TopBottom.hdf5"
    fsm.loadClassificationTopDown(
        classificationFile, data[NPname], NPname, atomMask=mask
    )
    fsm.loadClassificationTopDown(
        "../minimized.hdf5", data[NPname], NPname, atomMask=mask
    )
    # figsize = numpy.array([3.8, 3]) * 4

    fig, ax = plt.subplots(figsize=(12, 8))
    plt.rcParams.update({"font.size": 21})
    ax = fsm.HistoMaker_only(
        ax,
        data[NPname],
        positions=[0, 1, 2, 9, 8, 3, 7, 5, 6, 4],
        barWidth=0.40,  # sia ideal che 673
        barSpace=0.05,
        # barWidth=0.45,
        # barSpace=0.01,
    )
    plt.show()
    histo673 = data[NPname][673]
    histo300 = data[NPname][300]
    # numpy.savetxt(f"histo673_{NP}", histo673)
    # numpy.savetxt(f"histo300_{NP}", histo300)
    #fig.savefig(f"histo300{NPname}.png", bbox_inches="tight", pad_inches=0, dpi=300)
# %%
