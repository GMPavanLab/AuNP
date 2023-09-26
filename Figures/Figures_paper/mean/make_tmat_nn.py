#%%
import figureSupportModule as fsm
import numpy
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import seaborn
import h5py

#%%
Temps = [300, 673]


def getFiguresData(np):
    data = {}
    data[np] = dict()
    # data = fsm.pcaLoaderBottomUp(f"../{np}pca.hdf5", fsm.trajectorySlice)
    # fsm.loadClassificationBottomUp(
    # f"../{np}classifications.hdf5", data, np, fsm.trajectorySlice
    # )
    fsm.loadClassificationTopDown(f"../../../{np}TopBottom.hdf5", data[np], np, atomMask=mask)

    for T in Temps:
        fsm.addTmatTD(data[np][T])
        fsm.addTmatTDNN(data[np][T])
    fsm.loadClassificationTopDown
    return data


def getFullDataStrided(np, window):
    data = {}
    # for NPname in ["2"]:
    data[np] = dict()
    # data = fsm.pcaLoaderBottomUp(f"../{np}pca.hdf5", fsm.trajectorySlice)
    # fsm.loadClassificationBottomUp(
    # f"../{np}classifications.hdf5", data, np, fsm.trajectorySlice
    # )
    fsm.loadClassificationTopDown(
        f"../../../{np}TopBottom.hdf5",
        data[np],
        np,
        TimeWindow=window,
        atomMask=mask,
    )

    for T in Temps:
        fsm.addTmatTD(data[np][T])
        fsm.addTmatTDNN(data[np][T])
    fsm.loadClassificationTopDown
    return data


def makeTmats(
    dataDict,
    tmatAddr,
    legendNames,
    reordering,
    np,
    figsize=numpy.array([1.5, 1]) * 10,
    zoom=0.1,
):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=figsize, dpi=300)
    for i, T in enumerate(Temps):
        # axes[i].set_aspect(1)
        axes[i].set_title(f"{T} K")
        # continue
        # qui
        tmat = dataDict[np][T][tmatAddr]
        mask = tmat == 0
        annots = True
        seaborn.heatmap(
            tmat,
            linewidths=1,
            ax=axes[i],
            annot=annots,
            mask=mask,
            # xticklabels=False,
            # yticklabels=False,
            xticklabels=fsm.topDownLabels,
            yticklabels=fsm.topDownLabels,
            square=True,
            cmap="rocket_r",
            cbar=False,
            annot_kws=dict(
                weight="bold",
            ),
        )

        fsm.decorateTmatWithLegend(legendNames, reordering, axes[i], zoom=zoom)
    return fig


def calculateTransitions(tmat, time):
    n = tmat.shape[0]
    rates = numpy.zeros_like(tmat)
    for i in range(n):
        for j in range(n):
            if i != j and tmat[i, j] != 0:
                rates[i, j] = 1.0 / (tmat[i, j] / time)
    return rates


def transitionsFigures(
    dataDict1,
    time1,
    dataDict2,
    time2,
    tmatAddr,
    legendNames,
    reordering,
    np,
    figsize=numpy.array([2, 3]) * 10,
    zoom=0.02,
    tcalc=calculateTransitions,
):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize, dpi=300)
    for j, (myDict, time) in enumerate([(dataDict1, time1), (dataDict2, time2)]):
        for i, T in enumerate(Temps):
            ax = axes[i, j]
            tmat = tcalc(myDict[np][T][tmatAddr], time)
            mask = tmat == 0
            annots = True  # fsm.getCompactedAnnotationsForTmat_percent(tmat)
            seaborn.heatmap(
                tmat,
                linewidths=0.1,
                ax=ax,
                annot=annots,
                mask=mask,
                square=True,
                cmap="rocket",
                vmin=2,
                vmax=60,
                cbar=False,
                xticklabels=fsm.topDownLabels,
                yticklabels=fsm.topDownLabels,
                annot_kws=dict(
                    weight="bold",
                ),
            )
            fsm.decorateTmatWithLegend(legendNames, reordering, ax, zoom=zoom)
    return fig


def transitionsFigures2(
    dataDict,
    time1,
    tmatAddr,
    legendNames,
    reordering,
    np,
    figsize=numpy.array([2, 3]) * 10,
    zoom=0.02,
    tcalc=calculateTransitions,
):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize, dpi=300)
    for j, (myDict, time) in enumerate([(dataDict, time1)]):
        for i, T in enumerate(Temps):
            ax = axes[i]
            tmat = tcalc(myDict[np][T][tmatAddr], time)
            mask = tmat == 0
            annots = True  # fsm.getCompactedAnnotationsForTmat_percent(tmat)
            seaborn.heatmap(
                tmat,
                linewidths=0.1,
                ax=ax,
                annot=annots,
                mask=mask,
                square=True,
                cmap="rocket",
                vmin=2,
                vmax=60,
                cbar=False,
                xticklabels=fsm.topDownLabels,
                yticklabels=fsm.topDownLabels,
                annot_kws=dict(
                    weight="bold",
                ),
            )
            fsm.decorateTmatWithLegend(legendNames, reordering, ax, zoom=zoom)
    return fig


def plotEquilibrium(dataDict: dict, T: int, Class: str, colorMap: list, ax=None):
    axis = plt if ax is None else ax
    for i in range(len(colorMap)):
        t = numpy.count_nonzero(dataDict[np][T][Class].references == i, axis=-1)
        axis.plot(range(t.shape[0]), t, color=colorMap[i])


def plotEquilibriumWithMean(
    dataDict: dict, T: int, Class: str, colorMap: list, ax=None
):
    from pandas import DataFrame

    axis = plt if ax is None else ax
    for i in range(len(colorMap)):
        t = numpy.count_nonzero(dataDict[np][T][Class].references == i, axis=-1)
        df = DataFrame(t)
        axis.plot(range(t.shape[0]), t, color=colorMap[i], alpha=0.5)
        axis.plot(range(t.shape[0]), df.rolling(10).mean(), color=colorMap[i])


def stackEquilibrium(
    dataDict: dict, T: int, Class: str, colorMap: list, ax=None, reorder=None
):
    axis = plt if ax is None else ax
    t = numpy.zeros((len(colorMap), dataDict[T][Class].references.shape[0]), dtype=int)
    for i in range(len(colorMap)):
        t[i] = numpy.count_nonzero(dataDict[T][Class].references == i, axis=-1)
    colors = colorMap
    if reorder:
        t[:] = t[reorder]
        colors = colorMap[reorder]
    axis.stackplot(range(t.shape[1]), t, colors=colors)


def plotEquilibriumSingle(
    dataDict: dict,
    T: int,
    classNum: int,
    Class: str,
    np: int,
    color: list,
    label: str = None,
    ax=None,
):
    axis = plt if ax is None else ax
    # qui
    t = numpy.count_nonzero(dataDict[np][T][Class].references == classNum, axis=-1)
    axis.plot(range(t.shape[0]), t, color=color, label=label)


def figureEquilibrium(
    np,
    dataDict: dict,
    Class: str,
    cmap: list,
    figkwargs: dict,
    labels: list,
    reorder: list = None,
    vline: float = None,
):
    tdNum = len(cmap)
    fig, axes = plt.subplots(
        nrows=tdNum, ncols=2, sharex="col", sharey="row", **figkwargs
    )
    reordering = reorder if reorder else range(tdNum)

    for i, T in enumerate(Temps):
        for j in range(tdNum):
            jj = reordering[j]
            plotEquilibriumSingle(
                dataDict, T, j, Class, cmap[j], labels[jj], axes[jj, i]
            )
            axes[jj, i].legend(bbox_to_anchor=(1, 1), loc="right", framealpha=1)
            if i == 0:
                axes[jj, i].yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
                axes[jj, i].set_ylabel("Number of atoms")
            if jj == (tdNum - 1):
                t = [i for i in range(0, 1001, 500)]
                axes[jj, i].set_xticks(t, [i / 500 for i in t])
                axes[jj, i].set_xlabel("md time [$\mu$s]")
            if jj == 0:
                axes[jj, i].set_title(f"{T} K")
            if vline:
                axes[jj, i].axvline(vline, color="k", linestyle="--")
    fig.align_ylabels(axes[:, 0])
    return fig


#%%
if __name__ == "__main__":
    np = ["01", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
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
    for (n, t) in zip(np, trajs):
        with h5py.File(f"../../../{n}.hdf5", "r") as workFile:
            mask = workFile[f"Trajectories/{t}/Trajectory"][0, :, 2] > 84
        values_nn = getFiguresData(n)
        figsize = numpy.array([99, 240]) * 0.1
        for i, (tmatAddr, legendNames, npTitle, reordering) in enumerate(
            [
                # ("tmatBUNN", "bottomUp", "Bottom-Up", fsm.bottomReordering_r),
                ("tmatTDNN", "topDown", "Top-Down", range(10)),
            ]
        ):
            fig = makeTmats(
                values_nn,
                tmatAddr,
                legendNames,
                reordering,
                figsize=figsize,
                np=n,
                zoom=0.04,
            )

            fig.suptitle(f"NN Matrix for {n}", fontsize=16, y=1)
            fig.set_layout_engine("tight")
            fig.savefig(
                f"matrix_not_normalized_{n}.png", bbox_inches="tight", pad_inches=0, dpi=300
            )

        figsize = numpy.array([200, 230]) * 0.05
# %%
