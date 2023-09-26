#%%
import matplotlib.pyplot as plt
import h5py
from matplotlib.image import imread
import figureSupportModule as fsm

import numpy

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


def addNPImages(axes, data, NP):
    clusters0K = []
    NClasses = 10
    for T in ["Ideal", 300, 673]:
        p = [1.0]
        if T == "Ideal":
            ideal = data[T]["ClassTD"].references[0]
            clusters0K += [
                c for c in range(3, 10) if numpy.count_nonzero(ideal == c) > 0
            ]
        elif len(clusters0K) != 7:
            # print(clusters0K)
            clusterCountNotID = numpy.zeros(
                (data[T]["ClassTD"].references.shape[0]), dtype=float
            )
            clusterCountID = numpy.zeros(
                (data[T]["ClassTD"].references.shape[0]), dtype=float
            )
            for c in range(3, 10):
                if c in clusters0K:
                    clusterCountID += numpy.count_nonzero(
                        data[T]["ClassTD"].references == c, axis=-1
                    )
            surfClusters = clusterCountID + clusterCountNotID
            clusterCountID /= surfClusters
            clusterCountNotID /= surfClusters
            p[0] = numpy.mean(clusterCountID)
            p += [numpy.mean(clusterCountNotID)]
        axes[f"np{T}"].imshow(imread(f"../nanop/{NP}_{T}-Topdown.png"))
        title = f"{T}\u2009K" if T != "Ideal" else "Starting Frame"
        plt.rcParams.update({"font.size": 21})
        axes[f"np{T}"].set_title(title)
        # pieax = axes[f"np{T}"].inset_axes([0.7, -0.2, 0.3, 0.3])
        axes[f"np{T}"].axis("off")


for (NPname, t) in zip(np, trajs):
    with h5py.File(f"../../../{NPname}.hdf5", "r") as workFile:
        mask = workFile[f"Trajectories/{t}/Trajectory"][0, :, 2] > 84
    data = {}
    # for NPname in ["10"]:
    data[NPname] = dict()
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
    figsize = numpy.array([4, 7.6]) * 4
    fig, axes = fsm.makeLayout7(
        labelsOptions=dict(fontsize=0), temps=[300, 673], figsize=figsize, dpi=300
    )
    addNPImages(axes, data[NP], NP)
    
    for T in [300, 673]:
        # print(data[NP][T].keys())

        fsm.AddTmatsAndChord5_6_7(
            axes,
            data[NPname][T],
            T,
            zoom=0.015,
            cbarAx=None if T != 673 else axes["tmatCMAP"],
            tmatOptions=dict(
                # cmap="vlag_r",
                #annot=fsm.getCompactedAnnotationsForTmat_percent(data[NP][T]["tmatTD"]),
                fmt="s",
                linewidth=1,
                cbar_kws={} if T != 673 else {"label": "Probability"},
            ),
            chordOptions=dict(
                visualizationScale=0.9,
                labels=fsm.topDownLabels,
                labelpos=1.15,
                labelskwargs = dict(),
            ),
        )
    fsm.HistoMaker(
        axes["Histo"],
        data[NPname],
        positions=[0, 1, 2, 9, 8, 3, 7, 5, 6, 4],
        barWidth=0.22,
        barSpace=0.05,
    )
    t673 = data[NPname][673]["tmatTD"]
    t300 = data[NPname][300]["tmatTD"]
    numpy.savetxt(f"tmat673_{NP}.txt", t673)
    numpy.savetxt(f"tmat300_{NP}.txt", t300)

# %%
