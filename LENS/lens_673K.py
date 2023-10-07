#%%
import SOAPify.HDF5er as HDF5er
import SOAPify
import h5py
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.signal import savgol_filter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from multiprocessing.pool import ThreadPool as Pool
import seaborn
from seaborn import kdeplot

cutoff = 4.88
wantedTrajectory = slice(0, None, 1)
trajFileNames = ["01.hdf5","2.hdf5", "3.hdf5", "4.hdf5", "5.hdf5","6.hdf5","7.hdf5","8.hdf5","9.hdf5","10.hdf5",]
trajAddresses = [
    "/Trajectories/frame_01-SV_4835-SL_29282-T_300",
    "/Trajectories/frame_2-SV_12607-SL_11090-T_673/",
    "/Trajectories/frame_3-SV_11574-SL_28497-T_673/",
    "/Trajectories/frame_4-SV_21944-SL_17531-T_673/",
    "/Trajectories/frame_5-SV_9081-SL_31682-T_673/",
    "/Trajectories/frame_6-SV_3530-SL_10774-T_673/",
    "/Trajectories/frame_7-SV_17232-SL_25932-T_673/",
    "/Trajectories/frame_8-SV_13579-SL_31324-T_673/",
    "/Trajectories/frame_9-SV_8277-SL_3001-T_673/",
    "/Trajectories/frame_10-SV_5675-SL_26792-T_673/",
    
]


def signaltonoise(a: np.array, axis, ddof):
    """Given an array, retunrs its signal to noise value of its compontens"""
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return 20 * np.log10(abs(np.where(sd == 0, 0, m / sd)))

def transposeAndFlatten(x, /):
    trans = np.transpose(x)
    trans_fl = np.reshape(trans, np.shape(trans)[0] * np.shape(trans)[1])
    return trans_fl

def prepareData(x, /):
    """prepares an array from shape (atom,frames) to  (frames,atom)"""
    shape = x.shape
    toret = np.empty((shape[1], shape[0]), dtype=x.dtype)
    for i, atomvalues in enumerate(x):
        toret[:, i] = atomvalues
    return toret



def prepareData(x, /):
    """prepares an array from shape (atom,frames) to  (frames,atom)"""
    shape = x.shape
    toret = np.empty((shape[1], shape[0]), dtype=x.dtype)
    for i, atomvalues in enumerate(x):
        toret[:, i] = atomvalues
    return toret


# classyfing by knoing the min/max of the clusters
def classifying(x, classDict):
    toret = np.ones_like(x, dtype=int) * (len(classDict) - 1)
    # todo: sort  by max and then classify
    minmax = [[cname, data["max"]] for cname, data in data_KM.items()]
    minmax = sorted(minmax, key=lambda x: -x[1])
    # print(minmax)
    for cname, myMax in minmax:
        toret[x < myMax] = int(cname)
    return toret



def transposeAndFlatten(x, /):
    trans = np.transpose(x)
    trans_fl = np.reshape(trans, np.shape(trans)[0] * np.shape(trans)[1])
    return trans_fl

def prepareData(x, /):
    """prepares an array from shape (atom,frames) to  (frames,atom)"""
    shape = x.shape
    toret = np.empty((shape[1], shape[0]), dtype=x.dtype)
    for i, atomvalues in enumerate(x):
        toret[:, i] = atomvalues
    return toret


# classyfing by knoing the min/max of the clusters
def classifying(x, classDict):
    toret = np.ones_like(x, dtype=int) * (len(classDict) - 1)
    # todo: sort  by max and then classify
    minmax = [[cname, data["max"]] for cname, data in data_KM.items()]
    minmax = sorted(minmax, key=lambda x: -x[1])
    # print(minmax)
    for cname, myMax in minmax:
        toret[x < myMax] = int(cname)
    return toret

for trajFileName, trajAddress in zip(trajFileNames, trajAddresses):
    with h5py.File(f"/home/matteo/Scrivania/AuNanop/Analisi/RepoForGold/{trajFileName}", "r") as trajFile:
        tgroup = trajFile[trajAddress]
        universe = HDF5er.createUniverseFromSlice(tgroup, wantedTrajectory)
        
        nAtoms = len(universe.atoms)
        neigCounts = SOAPify.analysis.listNeighboursAlongTrajectory(universe, cutOff=cutoff)
        LENS, nn, *_ = SOAPify.analysis.neighbourChangeInTime(neigCounts)
        
        fig, axes = plt.subplots(2, sharey=True)
        for i in range(4):
            axes[0].plot(LENS[i], label=f"Atom {i}")
            axes[1].plot(LENS[nAtoms - 1 - i], label=f"Atom {nAtoms-1-i}")

        for ax in axes:
            ax.legend()
            ax.set_title(trajFileName)

        plt.show()
        
        atom = nAtoms - 1
        savGolPrint = np.arange(LENS[atom].shape[0])
        window_length = 100
        polyorder = 2
        windowToUSE = slice(window_length // 2, -window_length // 2)
        filteredLENS = savgol_filter(
            LENS, window_length=window_length, polyorder=polyorder, axis=-1
        )



        dataToFit = transposeAndFlatten(filteredLENS[:, windowToUSE])
        print("minmax: ", np.min(dataToFit), np.max(dataToFit))
        bestnclusters = 3
# fixing the random state for reproducibility
        clusters = KMeans(bestnclusters, random_state=12345).fit_predict(
        dataToFit.reshape(-1, 1)
        )
        data_KM = {}
        for i in range(bestnclusters):
            data_KM[i] = {}
            data_KM[i]["elements"] = dataToFit[clusters == i]
            data_KM[i]["min"] = np.min(data_KM[i]["elements"])
            data_KM[i]["max"] = np.max(data_KM[i]["elements"])
        
        fig, axes = plt.subplots(
            1, 2, figsize=(6, 3), dpi=200, width_ratios=[3, 1.2], sharey=True
        )

        for flns in filteredLENS:
            axes[0].plot(
                savGolPrint[windowToUSE],
                flns[windowToUSE],
                color="k",
                linewidth=0.01,
        # alpha=0.9,
            )
        hist = axes[1].hist(
            dataToFit,
            alpha=0.7,
            color="gray",
            bins="doane",
            density=True,
            orientation="horizontal",
        )
        kde = kdeplot(
            y=dataToFit,
            bw_adjust=1.5,
            linewidth=2,
            color="black",
            # gridsize=500,
            gridsize=2 * (hist[1].shape[0] - 1),
            ax=axes[1],
        )
        axes[0].set_title(f'LENS trajectories of {trajFileName}')
        
        axes[0].set_xlim(0, len(savGolPrint))
        axes[1].set_title("Density of\nLENS values")
        height = np.max(hist[0])
        for cname, data in data_KM.items():
            # center = delta+data['min']
            bin_size = data["max"] - data["min"]
            for i, ax in enumerate(axes):
                ax.barh(
                    y=data["min"],
                    width=len(savGolPrint) if i == 0 else height,
                    height=bin_size,
                    align="edge",
                    alpha=0.25,
                    #color=palette[tab20],
                    # color=colors[idx_m],
                    linewidth=2,
                )
                ax.axhline(y=data["min"], c="red", linewidth=2, linestyle=":")
                if i == 1:
                    ax.text(
                        -0.25,
                        data["min"] + bin_size / 2.0,
                        f"c{cname}",
                        va="center",
                        ha="right",
                        fontsize=18,
                    )
        classifiedFilteredLENS = classifying(filteredLENS, data_KM)
        def export(wantedTrajectory):
    # as a function, so the ref universe should be garbage collected correctly
            with h5py.File(f"/home/matteo/Scrivania/AuNanop/Analisi/RepoForGold/{trajFileName}", "r") as trajFile, open(
                f"./673K/outClassifiedLENS_{trajFileName}_673K.xyz", "w"
            ) as xyzFile:
                from MDAnalysis.transformations import fit_rot_trans

                tgroup = trajFile[trajAddress]
                ref = HDF5er.createUniverseFromSlice(tgroup, [0])
                nAt= len(ref.atoms)
                ref.add_TopologyAttr("mass", [1] * nAt)
                # load antohter univer to avoid conatmination in the main one
                exportuniverse = HDF5er.createUniverseFromSlice(tgroup, wantedTrajectory)
                exportuniverse.add_TopologyAttr("mass", [1] * nAt)
                exportuniverse.trajectory.add_transformations(fit_rot_trans(exportuniverse, ref))
                HDF5er.getXYZfromMDA(
                    xyzFile,
                    exportuniverse,
                    # framesToExport=wantedTrajectory,
                    allFramesProperty='Origin="-40 -40 -40"',
                    LENS=prepareData(LENS),
                    filteredLENS=prepareData(filteredLENS),
                    proposedClassification=prepareData(classifiedFilteredLENS),
                )
                universe.trajectory


        export(wantedTrajectory)
# %%
