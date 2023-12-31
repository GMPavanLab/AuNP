#%%
# from matplotlib.gridspec import
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_rgb
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.image import imread
from mpl_toolkits.axisartist.axislines import AxesZero
import h5py, re, numpy, sys, seaborn
from string import ascii_lowercase as alph
from chorddiagram import ChordDiagram
from scipy.ndimage import gaussian_filter

# import colorsys
from SOAPify import (
    transitionMatrixFromSOAPClassificationNormalized as tmatMaker,
    transitionMatrixFromSOAPClassification as tmatMakerNN,
    SOAPclassification,
)

sys.path.insert(0, "../topDown")
from referenceMaker import (
    desiredReferenceOrderIco,
    desiredReferenceOrderTo,
    desiredReferenceOrderDh,
)

__reT = re.compile("T_([0-9]*)")


def getT(s):
    match = __reT.search(s)
    if match:
        return int(match.group(1))
    else:
        # this is a trick for loading data in createXYZ.py
        return "Ideal"


trajectorySlice = slice(10000, None, 10)
#%%
# ../bottomUp/ico309soap.hdf5
#

bottomUpcolorMapHex = [
    "#00b127",  # Faces
    "#e00081",  # Concave
    "#0086ba",  # 5foldedSS
    "#003ac6",  # Ico
    "#450055",  # Bulk
    "#3800a8",  # SubSurf
    "#bdff0e",  # Edges
    "#ffe823",  # Vertexes
]
bottomUpColorMap = numpy.array([to_rgb(k) for k in bottomUpcolorMapHex])

bottomUpLabels = [
    "Faces",  # 0
    "Concave",  # 1
    "5foldedSS",  # 2
    "Ico",  # 3
    "Bulk",  # 4
    "SubSurf",  # 5
    "Edges",  # 6
    "Vertexes",  # 7
]
bottomUpLabels_ordered = [
    "Ico",
    "Bulk",
    "SubSurf",
    "5foldedSS",
    "Concave",
    "Faces",
    "Edges",
    "Vertexes",
]
bottomReordering = [7, 6, 0, 1, 2, 5, 4, 3]
bottomReordering_r = bottomReordering[::-1]

topDownColorMapHex = [
    "#463088",  # b
    "#713CCD",  # ss
    "#9B50E2",  # ss'
    "#33a02c",  # c !
    "#8ee773",  # c'
    "#CD86EA",  # s
    "#ffff00",  # e !
    "#E6BAF2",  # e'
    "#F3DCF9",
    "#e31a1c",  # v' !
    # v
]
topDownColorMapHex2 = [
    "#708090",  # b
    "#a6cee3",  # ss
    "#1f78b4",  # ss'
    "#33a02c",  # c
    "#8ee773",  # c'
    "#e31a1c",  # s
    "#fdbf6f",  # e
    "#ff7f00",  # e'
    "#cab2d6",  # v'
    "#6a3d9a",  # v
]
topDownColorMap = numpy.array([to_rgb(k) for k in topDownColorMapHex])

topDownLabels = [
    "b",  # 0 -> 0
    "ss",  # 1 -> 1
    "ss'",  # 2 -> 2
    "c",  # 5 -> 3
    "c'",  # 3 -> 4
    "s",  # 4 -> 5
    "e",  # 6 -> 6
    "e'",  # 7 -> 7
    "v'",  # 9 -> 8
    "v",  # 8 -> 9
]
# ho cluster are ordered in fcluster
topDownLabelFound = [
    "b",  # 0 -> 0
    "ss",  # 1 -> 1
    "ss'",  # 2 -> 2
    "c'",  # 3 -> 4
    "s",  # 4 -> 5
    "c",  # 5 -> 3
    "e",  # 6 -> 6
    "e'",  # 7 -> 7
    "v",  # 8 -> 9
    "v'",  # 9 -> 8
]
topDownReordering = [0, 1, 2, 5, 3, 4, 6, 7, 9, 8]
# [0, 1, 2, 4, 5, 3, 6, 7, 9, 8]
# this is the returnn of [topDownLabels.index(k) for k in topDownLabelFound]
# applied on fcluster
topDownClusters = numpy.array(
    [
        9,
        6,
        7,
        5,
        5,
        2,
        1,
        1,
        0,
        0,
        0,
        0,
        8,
        7,
        7,
        7,
        7,
        5,
        5,
        5,
        5,
        2,
        1,
        1,
        0,
        9,
        9,
        7,
        6,
        6,
        8,
        8,
        4,
        7,
        3,
        3,
        4,
        4,
        5,
        5,
        1,
        1,
        2,
        1,
        0,
        0,
        0,
    ]
)

topDown_ihColorMap = list(
    seaborn.color_palette("Blues", n_colors=len(desiredReferenceOrderIco))
)[::-1]
topDown_toColorMap = list(
    seaborn.color_palette(
        "Purples",
        n_colors=len(desiredReferenceOrderTo),
    )
)[::-1]
topDown_dhColorMap = list(
    seaborn.color_palette("Greens", n_colors=len(desiredReferenceOrderDh))
)[::-1]

topDownFullColorMap = topDown_ihColorMap + topDown_toColorMap + topDown_dhColorMap


def getF(proposed: float, concurrentArray, F):
    avversary = F(concurrentArray)
    return F([proposed, avversary])


def getMin(proposed: float, concurrentArray):
    return getF(proposed, concurrentArray, numpy.min)


def getMax(proposed: float, concurrentArray):
    return getF(proposed, concurrentArray, numpy.max)


def pcaLoaderBottomUp(filename: str, TimeWindow: slice = slice(None)):
    dataContainer = dict()
    dataContainer["xlims"] = [numpy.finfo(float).max, numpy.finfo(float).min]
    dataContainer["ylims"] = [numpy.finfo(float).max, numpy.finfo(float).min]
    with h5py.File(filename, "r") as f:
        pcas = f["/PCAs/ico309-SV_18631-SL_31922-T_300"]
        for k in pcas:
            dataContainer["nat"] = pcas[k].shape[1]
            T = getT(k)
            dataContainer[T] = dict(pca=pcas[k][TimeWindow, :, :2].reshape(-1, 2))
            dataContainer["xlims"][0] = getMin(
                dataContainer["xlims"][0], dataContainer[T]["pca"][:, 0]
            )
            dataContainer["xlims"][1] = getMax(
                dataContainer["xlims"][1], dataContainer[T]["pca"][:, 0]
            )
            dataContainer["ylims"][0] = getMin(
                dataContainer["ylims"][0], dataContainer[T]["pca"][:, 1]
            )
            dataContainer["ylims"][1] = getMax(
                dataContainer["ylims"][1], dataContainer[T]["pca"][:, 1]
            )

    return dataContainer


def loadClassification(
    classificationFile: str,
    dataContainer: dict,
    NPname: str,
    ClassificationPlace: str,
    ClassNameToFind: str,
    ClassNameToSave: str,
    TimeWindow: slice = slice(None),
):
    with h5py.File(classificationFile, "r") as f:
        Simulations = f[ClassificationPlace]
        for k in Simulations:
            if NPname in k:
                T = getT(k)
                if T not in dataContainer:
                    dataContainer[T] = dict()
                dataContainer[T][ClassNameToSave] = SOAPclassification(
                    [], Simulations[k][ClassNameToFind][TimeWindow], bottomUpLabels
                )


def loadClassificationBottomUp(
    classificationFile: str,
    dataContainer: dict,
    NPname: str,
    TimeWindow: slice = slice(None),
):
    loadClassification(
        classificationFile,
        dataContainer,
        NPname,
        ClassificationPlace="/Classifications/ico309-SV_18631-SL_31922-T_300",
        ClassNameToFind="labelsNN",
        ClassNameToSave="ClassBU",
        TimeWindow=TimeWindow,
    )


def loadClassificationTopDown(
    classificationFile: str,
    data: dict,
    NPname: str,
    atomMask: slice = slice(None),
    TimeWindow: slice = slice(None),
):

    ClassificationPlace = "Classifications/icotodh"
    ClassNameToSave = "ClassTD"
    with h5py.File(classificationFile, "r") as distFile:
        ClassG = distFile[ClassificationPlace]
        for k in ClassG:
            if NPname in k:
                T = getT(k)
                if T not in data:
                    data[T] = dict()
                classification = ClassG[k][TimeWindow][:, atomMask]
                # this
                clusterized = topDownClusters[classification]
                data[T][ClassNameToSave] = SOAPclassification(
                    [], clusterized, topDownLabels
                )


#%% Layouts

__titleDict = dict(
    fontdict=dict(weight="bold", horizontalalignment="right"),
    loc="left",
)


def makeLayout1(labelsOptions=dict(), **figkwargs):
    fig = plt.figure(**figkwargs)
    mainGrid = fig.add_gridspec(2, 4)

    axes = dict(
        soapAx=fig.add_subplot(mainGrid[0, 0]),
        idealAx=fig.add_subplot(mainGrid[0, 1]),
        idealSlicedAx=fig.add_subplot(mainGrid[0, 2]),
        legendAx=fig.add_subplot(mainGrid[0, 3]),
    )
    axes.update(createSimulationFigs(mainGrid[1, :].subgridspec(1, 4), fig, name="300"))

    titleDict = __titleDict.copy()
    titleDict["fontdict"].update(labelsOptions)
    for i, ax in enumerate(
        [
            axes["soapAx"],
            axes["idealAx"],
            axes["idealSlicedAx"],
            axes["legendAx"],
            axes["img300Ax"],
            axes["pca300Ax"],
            axes["pFES300Ax"],
            axes["tmat300Ax"],
        ]
    ):
        ax.set_title(alph[i] + ("  " if i in [5, 6] else ""), **titleDict)
    return fig, axes


def makeLayout2(labelsOptions=dict(), **figkwargs):
    fig = plt.figure(**figkwargs)
    mainGrid = fig.add_gridspec(2, 4)

    axes = dict()
    axes.update(createSimulationFigs(mainGrid[0, :].subgridspec(1, 4), fig, name="400"))
    axes.update(createSimulationFigs(mainGrid[1, :].subgridspec(1, 4), fig, name="500"))
    titleDict = __titleDict.copy()
    titleDict["fontdict"].update(labelsOptions)
    for i, ax in enumerate(
        [
            axes["img400Ax"],
            axes["pca400Ax"],
            axes["pFES400Ax"],
            axes["tmat400Ax"],
            axes["img500Ax"],
            axes["pca500Ax"],
            axes["pFES500Ax"],
            axes["tmat500Ax"],
        ]
    ):
        ax.set_title(alph[i] + ("  " if i in [1, 2, 5, 6] else ""), **titleDict)
    return fig, axes


def makeLayout3(labelsOptions=dict(), **figkwargs):
    axes = dict()
    fig = plt.figure(**figkwargs)
    mainGrid = fig.add_gridspec(nrows=2, ncols=4, width_ratios=[0.6, 0.1, 1, 1])
    NPGrid = mainGrid[:, 0].subgridspec(2, 1, height_ratios=[1, 1])
    graphSUPGrid = mainGrid[0, 2:].subgridspec(2, 2, height_ratios=[1, 0.1])
    graphINFGrid = mainGrid[1, 2:].subgridspec(2, 2, height_ratios=[1, 0.1])
    taxes = dict(
        npt=fig.add_subplot(mainGrid[0, 0]),
        npc=fig.add_subplot(mainGrid[1, 0]),
    )
    axes["NPTime"] = fig.add_subplot(NPGrid[0])
    axes["NPClasses"] = fig.add_subplot(NPGrid[-1])
    axes["NPcbar"] = axes["NPTime"].inset_axes([0.1, -0.05, 0.8, 0.05])
    # fig.add_subplot(mainGrid[:, 1])
    for g, n in [
        (graphSUPGrid[0, 0], "followGraph"),
        (graphSUPGrid[0, 1], "graphT300"),
        (graphINFGrid[0, 0], "graphT400"),
        (graphINFGrid[0, 1], "graphT500"),
    ]:
        axes[n] = fig.add_subplot(
            g  # , sharey=axes["followGraph"] if n != "followGraph" else None
        )
    titleDict = __titleDict.copy()
    titleDict["fontdict"].update(labelsOptions)
    spacer = {
        0: "",
        1: "",
        3: " " * 4,
        5: " " * 4,
        4: " " * 17,
        2: " " * 17,
    }
    for i, ax in enumerate(
        [
            taxes["npt"],
            taxes["npc"],
            axes["followGraph"],
            axes["graphT300"],
            axes["graphT400"],
            axes["graphT500"],
        ]
    ):
        ax.set_title(alph[i] + spacer[i], **titleDict)
    for ax in taxes:
        taxes[ax].axis("off")
    return fig, axes


def makeLayout5(labelsOptions=dict(), **figkwargs):
    axes = dict()
    fig = plt.figure(**figkwargs)
    mainGrid = fig.add_gridspec(
        nrows=4,
        ncols=4,
        # width_ratios=[0.8, 1, 1, 1],
        # height_ratios=[1, 2, 2, 2]
        width_ratios=[0.8, 1, 1, 2],
        height_ratios=[1, 2, 2, 2],
        hspace=0.3,
        wspace=0.3,
    )
    legendGrid = mainGrid[0, :].subgridspec(1, 2, wspace=0.0)
    axes["dendro"] = fig.add_subplot(legendGrid[0])
    axes["legend"] = fig.add_subplot(legendGrid[1])
    npGrid = mainGrid[1:, 0].subgridspec(4, 1)
    # tmatGrid = mainGrid[3, 1:].subgridspec(1, 3)
    tmatGrid = mainGrid[3, 1:].subgridspec(1, 5, width_ratios=[1, 1, 0.05, 0.05, 0.05])
    chordGrid = mainGrid[2, 1:].subgridspec(1, 2)
    axes[f"npIdeal"] = fig.add_subplot(npGrid[0])
    for i, T in enumerate([300, 673]):
        axes[f"np{T}"] = fig.add_subplot(npGrid[i + 1])
        axes[f"tmat{T}"] = fig.add_subplot(tmatGrid[i])
        axes[f"chord{T}"] = fig.add_subplot(chordGrid[i])
    axes[f"tmatCMAP"] = fig.add_subplot(tmatGrid[3])
    axes["Histo"] = fig.add_subplot(mainGrid[1, 1:])
    taxes = dict(
        nps=fig.add_subplot(mainGrid[1:, 0]),
        chords=fig.add_subplot(mainGrid[2, 1:]),
        tmats=fig.add_subplot(mainGrid[3, 1:]),
    )
    titleDict = __titleDict.copy()
    titleDict["fontdict"].update(labelsOptions)
    for i, ax in enumerate(
        [
            axes["dendro"],
            taxes["nps"],
            axes["Histo"],
            taxes["chords"],
            taxes["tmats"],
        ]
    ):
        ax.set_title(alph[i], **titleDict)
    for ax in taxes:
        taxes[ax].axis("off")

    return fig, axes


def makeLayout6and7(labelsOptions=dict(), temps=[300, 400, 500], **figkwargs):
    axes = dict()
    fig = plt.figure(**figkwargs)
    mainGrid = fig.add_gridspec(
        nrows=3,
        ncols=2,
        height_ratios=[0.8, 1.2, 2],
        width_ratios=[3, 1.2],
        wspace=0.2,
    )
    npGrid = mainGrid[0, 0].subgridspec(1, 3)
    tmatGrid = mainGrid[2, 0].subgridspec(1, 3, width_ratios=[1, 1, 0.001])
    chordGrid = mainGrid[:, 1].subgridspec(2, 1)
    axes[f"npIdeal"] = fig.add_subplot(npGrid[0])
    for i, T in enumerate(temps):
        axes[f"np{T}"] = fig.add_subplot(npGrid[i + 1])
        axes[f"tmat{T}"] = fig.add_subplot(tmatGrid[i])
        axes[f"chord{T}"] = fig.add_subplot(chordGrid[i])
    axes[f"tmatCMAP"] = fig.add_subplot(tmatGrid[2])
    axes[f"nps"] = fig.add_subplot(npGrid[:])
    axes[f"tmats"] = fig.add_subplot(tmatGrid[:])
    axes[f"chords"] = fig.add_subplot(chordGrid[:])
    axes["Histo"] = fig.add_subplot(mainGrid[1, 0])
    titleDict = __titleDict.copy()
    titleDict["fontdict"].update(labelsOptions)
    for i, ax in enumerate(
        [
            axes["nps"],
            axes["Histo"],
            axes["tmats"],
            axes["chords"],
        ]
    ):
        ax.set_title(alph[i], **titleDict)
    for ax in [axes["nps"], axes["tmats"], axes["chords"]]:
        ax.axis("off")
    return fig, axes


#%%
def addPseudoFes(tempData, bins=300, rangeHisto=None):
    hist, xedges, yedges = numpy.histogram2d(
        tempData["pca"][:, 0],
        tempData["pca"][:, 1],
        bins=bins,
        range=rangeHisto,
        density=True,
    )
    tempData["pFES"] = -numpy.log(hist.T)
    tempData["pFES"] -= numpy.min(tempData["pFES"])
    tempData["pFESLimitsX"] = (xedges[:-1] + xedges[1:]) / 2
    tempData["pFESLimitsY"] = (yedges[:-1] + yedges[1:]) / 2


def createSimulationFigs(grid, fig, name=""):
    toret = dict()
    pcaFESGrid = grid[1:3].subgridspec(1, 4, width_ratios=[1, 1, 0.1, 0.1])
    toret[f"img{name}Ax"] = fig.add_subplot(grid[0])
    toret[f"pca{name}Ax"] = fig.add_subplot(pcaFESGrid[0], axes_class=AxesZero)
    toret[f"pFES{name}Ax"] = fig.add_subplot(pcaFESGrid[1], axes_class=AxesZero)
    toret[f"cbarFes{name}Ax"] = fig.add_subplot(pcaFESGrid[2])
    toret[f"tmat{name}Ax"] = fig.add_subplot(grid[3])
    return toret


#%%


def decorateTmatWithLegend(
    legendName,
    legendReordering,
    ax,
    zoom=0.025,
    offset=1,
    horizontal=True,
    vertical=True,
):
    n = len(legendReordering)
    for i, l in enumerate(legendReordering):
        img = OffsetImage(imread(f"{legendName}{l:04}.png"), zoom=zoom)
        if vertical:
            ab = AnnotationBbox(
                img,
                (0, i + 0.55),
                xybox=(-offset, i + 0.55),
                frameon=False,
                xycoords="data",
                box_alignment=(0.55, 0.55),
            )
            ax.add_artist(ab)
        if horizontal:
            ab = AnnotationBbox(
                img,
                (i + 0.51, 0),
                xybox=(i + 0.51, n + offset),
                frameon=False,
                xycoords="data",
                box_alignment=(0.51, 0.51),
            )
            ax.add_artist(ab)


def decorateTmatWithLegend2(
    legendName,
    legendReordering,
    ax,
    zoom=0.03,
    offset=0.25,
    horizontal=True,
    vertical=True,
):
    n = len(legendReordering)
    for i, l in enumerate(legendReordering):
        img = OffsetImage(imread(f"{legendName}{l:04}.png"), zoom=zoom)
        if vertical:
            ab = AnnotationBbox(
                img,
                (0, i + 0.5),
                xybox=(-offset, i + 0.5),
                frameon=False,
                xycoords="data",
                #  box_alignment=(0.5, 0.5),
            )
            ax.add_artist(ab)
        if horizontal:
            ab = AnnotationBbox(
                img,
                (i + 0.5, 0),
                xybox=(i + 0.5, n + offset),
                frameon=False,
                xycoords="data",
                # box_alignment=(0.5, 0.5),
            )
            ax.add_artist(ab)


def getCompactedAnnotationsForTmat_percent(tmat) -> list:
    """
    Returns a list of compacted annotations for a given tmat.
    """
    annot = list(numpy.empty(tmat.shape, dtype=str))
    # annot=numpy.chararray(tmat.shape, itemsize=5)
    for row in range(tmat.shape[0]):
        annot[row] = list(annot[row])
        for col in range(tmat.shape[1]):
            if tmat[row, col] < 0.01:
                annot[row][col] = f"<1"
            elif tmat[row, col] > 0.99:
                annot[row][col] = f">99"
            else:
                annot[row][col] = f"{100*tmat[row,col]:.0f}"
    return annot


def addTmatBU(tempData):
    tempData["tmatBU"] = tmatMaker(tempData["ClassBU"])[bottomReordering_r][
        :, bottomReordering_r
    ]


def addTmatBUNN(tempData):
    tempData["tmatBUNN"] = tmatMakerNN(tempData["ClassBU"])[bottomReordering_r][
        :, bottomReordering_r
    ]


def addTmatTD(tempData):
    tempData["tmatTD"] = tmatMaker(tempData["ClassTD"])


def addTmatTD_everyn(tempData):
    tempData["tmatTD"] = tmatMaker(tempData["ClassTD"], 100)


def addTmatTDNN_everyn(tempData):
    tempData["tmatTDNN"] = tmatMakerNN(tempData["ClassTD"], 10)


def addTmatTDNN(tempData):
    tempData["tmatTDNN"] = tmatMakerNN(tempData["ClassTD"])


def AddTmats5_6_7(
    axesdict,
    data,
    T,
    zoom=0.01,
    cbarAx=None,
    tmatOptions=dict(),
):
    reorder = list(range(10))  # [0, 1, 2, 5, 3, 4, 6, 7, 9, 8]
    # tmat= data["tmatTD"]
    tmat = data[T]["tmatTD"]
    mask = tmat == 0

    tmatOpts = dict(
        ax=axesdict[f"tmat{T}"],
        annot=None,
        mask=mask,
        square=True,
        cmap="rocket_r",
        vmax=1,
        vmin=0,
        cbar=cbarAx is not None,
        cbar_ax=cbarAx,
        xticklabels=topDownLabels,
        yticklabels=topDownLabels,
    )
    tmatOpts.update(tmatOptions)
    seaborn.heatmap(
        tmat,
        **tmatOpts,
    )
    decorateTmatWithLegend("topDown", reorder, axesdict[f"tmat{T}"], zoom=zoom)


def AddTmatsAndChord5_6_7(
    axesdict, data, T, zoom=0.01, cbarAx=None, tmatOptions=dict(), chordOptions=dict()
):
    reorder = list(range(10))  # [0, 1, 2, 5, 3, 4, 6, 7, 9, 8]
    tmat = data["tmatTD"]
    mask = tmat == 0

    tmatOpts = dict(
        ax=axesdict[f"tmat{T}"],
        annot=None,
        mask=mask,
        square=True,
        cmap="rocket_r",
        # vmax=tmat.values.max(),
        # vmin=0,
        cbar=cbarAx is not None,
        cbar_ax=cbarAx,
        xticklabels=topDownLabels,
        yticklabels=topDownLabels,
        annot_kws={"size": 5},
    )
    tmatOpts.update(tmatOptions)
    seaborn.heatmap(
        tmat,
        **tmatOpts,
    )
    decorateTmatWithLegend("topDown", reorder, axesdict[f"tmat{T}"], zoom=zoom)
    chordOpts = dict(
        colors=topDownColorMap,
        ax=axesdict[f"chord{T}"],
        onlyFlux=True,
    )
    chordOpts.update(chordOptions)

    ChordDiagram(
        data["tmatTDNN"],
        **chordOpts,
    )
    for ax in [axesdict[f"chord{T}"], axesdict[f"tmat{T}"]]:
        ax.set_title(f"{T}\u2009K")


def HistoMaker(
    ax: plt.Axes,
    data: dict,
    barWidth: float = 0.15,
    barSpace: float = 0.05,
    arrowLenght: float = 0.25,
    positions=None,
):

    nHisto = len(topDownLabels)
    if positions is None:
        positions = range(nHisto)
    t = data["Ideal"]["ClassTD"]
    order = ["Ideal", 300, 673]
    # order = ["Ideal", 300, 400, 500]
    countMean = {T: numpy.zeros((nHisto), dtype=float) for T in data}
    countDev = {T: numpy.zeros((nHisto), dtype=float) for T in data}
    pos = {T: numpy.zeros((nHisto), dtype=float) for T in order}

    barsDim = 3 * barWidth + 2 * barSpace
    for c in range(nHisto):
        countMean["Ideal"][c] = numpy.count_nonzero(t.references[0] == c)
        countDev["Ideal"] = None
        dev = -barsDim / 2
        for T in order:
            pos[T][c] = positions[c] + dev
            dev += barWidth + barSpace
            if T == "Ideal":
                continue
            countC = numpy.count_nonzero(data[T]["ClassTD"].references == c, axis=-1)
            countMean[T][c] = numpy.mean(countC)
            countDev[T][c] = numpy.std(countC)
    ax.set_xticks(
        range(nHisto), [topDownLabels[positions.index(k)] for k in range(nHisto)]
    )
    styles = {
        "Ideal": dict(edgecolor="k", alpha=0.5),
        300: dict(hatch="\\\\\\", edgecolor="w"),
        # 673: dict(edgecolor="w"),
        673: dict(hatch="///", edgecolor="w"),
        # 500: dict(hatch="\\\\\\", edgecolor="w"),
    }
    for T in data:
        ax.bar(
            pos[T],
            countMean[T],
            yerr=countDev[T],
            width=barWidth,
            align="edge",
            color=topDownColorMap,
            **styles[T],
        )
    _, axylim = ax.get_ylim()
    scaleAx = 0.8
    ax.set_xlim(-barsDim * scaleAx, nHisto - 1 + barsDim * scaleAx)
    ax.set_ylabel("Mean Number of Atoms")
    for c in range(nHisto):
        if countMean["Ideal"][c] == 0:
            arrowXPos = pos["Ideal"][c] + barWidth / 2
            ax.annotate(
                "",
                xy=(arrowXPos, 0),
                xycoords="data",
                xytext=(arrowXPos, axylim * arrowLenght),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
            )
    legend_elements = []
    for k in styles:
        title = f"{k}\u2009K" if k != "Ideal" else "Starting frame"
        legend_elements.append(Patch(label=title, facecolor="#aaa", **styles[k]))

    ax.legend(handles=legend_elements)


#%%
pFEScmap = plt.cm.coolwarm_r.copy()
pFEScmap.set_over("w")


def plotTemperatureData(axesdict, T, data, xlims, ylims, zoom=0.01, smooth=0.0):
    pFES = data["pFES"]
    mymax = int(numpy.max(pFES[numpy.isfinite(pFES)]))
    if smooth > 0.0:
        t = numpy.array(pFES)
        t[numpy.isinf(t)] = numpy.max(t[numpy.isfinite(t)])
        pFES = gaussian_filter(t, sigma=smooth, order=0)

    # option for the countour lines
    countourOptions = dict(
        levels=10,
        colors="k",
        linewidths=0.1,
        zorder=2,
    )
    # the pca points
    axesdict[f"pca{T}Ax"].scatter(
        data["pca"][:, 0],
        data["pca"][:, 1],
        s=0.1,
        c=bottomUpColorMap[data["ClassBU"].references.reshape(-1)],
        alpha=0.5,
    )
    axesdict[f"pca{T}Ax"].contour(
        data["pFESLimitsX"],
        data["pFESLimitsY"],
        pFES,
        **countourOptions,
    )

    # pseudoFES representation
    pfesPlot = axesdict[f"pFES{T}Ax"].contourf(
        data["pFESLimitsX"],
        data["pFESLimitsY"],
        pFES,
        levels=10,
        cmap=pFEScmap,
        zorder=1,
        extend="max",
        vmax=mymax,
    )

    axesdict[f"pFES{T}Ax"].contour(
        data["pFESLimitsX"],
        data["pFESLimitsY"],
        pFES,
        **countourOptions,
    )

    cbar = plt.colorbar(
        pfesPlot,
        shrink=0.5,
        aspect=10,
        orientation="vertical",
        cax=axesdict[f"cbarFes{T}Ax"],
        label="$[k_BT]$",  # "_{\!_{" + self.sub + "}}"
    )
    mask = data["tmatBU"] == 0
    annots = getCompactedAnnotationsForTmat_percent(data["tmatBU"])
    seaborn.heatmap(
        data["tmatBU"],
        linewidths=0.1,
        ax=axesdict[f"tmat{T}Ax"],
        fmt="s",
        annot=annots,
        mask=mask,
        square=True,
        cmap="rocket_r",
        vmax=1,
        vmin=0,
        cbar=False,
        xticklabels=False,
        yticklabels=False,
        annot_kws=dict(
            weight="bold",
        ),
    )
    decorateTmatWithLegend(
        "bottomUp", bottomReordering_r, axesdict[f"tmat{T}Ax"], zoom=zoom
    )
    for ax in [axesdict[f"pca{T}Ax"], axesdict[f"pFES{T}Ax"]]:
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        for direction in ["left", "bottom"]:
            ax.axis[direction].set_axisline_style("-|>")

        for direction in ["right", "top"]:
            ax.axis[direction].set_visible(False)


#%%
def colorBarExporter(listOfColors, filename=None):
    from matplotlib.colors import ListedColormap
    from matplotlib.cm import ScalarMappable

    fig, ax = plt.subplots(figsize=(1, 6))
    ax.axis("off")
    fig.colorbar(
        ScalarMappable(cmap=ListedColormap(listOfColors)),
        cax=ax,
        orientation="vertical",
    )
    if filename:
        fig.savefig(fname=filename, bbox_inches="tight", pad_inches=0)


# colorBarExporter(topDownColorMap[::-1], "topDownCMAP.png")
# colorBarExporter(bottomUpColorMap[::-1], "bottomUPCMAP.png")
