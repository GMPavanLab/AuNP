

# %%
import matplotlib.pyplot as plt
import h5py
from matplotlib.image import imread
import figureSupportModule as fsm
import numpy as np  # Importa numpy correttamente

np_values = ["01", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

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

plt.rcParams.update({"font.size": 25})


def addNPImages(axes, data, NP):
    clusters0K = []
    NClasses = 10
    for T in ["Ideal", 673]:
        p = [1.0]
        if T == "Ideal":
            ideal = data[T]["ClassTD"].references[0]
            clusters0K += [
                c for c in range(3, 10) if np.count_nonzero(ideal == c) > 0
            ]
        elif len(clusters0K) != 7:
            clusterCountNotID = np.zeros(
                (data[T]["ClassTD"].references.shape[0]), dtype=float
            )
            clusterCountID = np.zeros(
                (data[T]["ClassTD"].references.shape[0]), dtype=float
            )
            for c in range(3, 10):
                if c in clusters0K:
                    clusterCountID += np.count_nonzero(
                        data[T]["ClassTD"].references == c, axis=-1
                    )
            surfClusters = clusterCountID + clusterCountNotID
            clusterCountID /= surfClusters
            clusterCountNotID /= surfClusters
            p[0] = np.mean(clusterCountID)
            p += [np.mean(clusterCountNotID)]
        axes[f"np{T}"].imshow(imread(f"../nanop/{NP}_{T}-Topdown.png"))
        title = f"{T}\u2009K" if T != "Ideal" else "Starting Frame"
        plt.rcParams.update({"font.size": 20})
        axes[f"np{T}"].set_title(title)
        axes[f"np{T}"].axis("off")


for (NPname, t) in zip(np_values, trajs):
    with h5py.File(f"../../../{NPname}.hdf5", "r") as workFile:
        mask = workFile[f"Trajectories/{t}/Trajectory"][0, :, 2] > 84
    data = {}
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
    figsize = np.array([4, 7]) * 4
    fig, axes = fsm.makeLayoutSI_onlyT(
        labelsOptions=dict(fontsize=0), temps=[673], figsize=figsize, dpi=300
    )
    addNPImages(axes, data[NPname], NPname)

    for T in [673]:
        fsm.AddTmatsAndChord_onlyT(
            axes,
            data[NPname][T],
            T,
            zoom=0.020,
            cbarAx=None,  # if T != 673 else axes["tmatCMAP"],
            tmatOptions=dict(
                fmt="s",
                linewidth=1,
                # cbar_kws={} if T != 673 else {"label": "Probability"},
            ),
            chordOptions=dict(
                visualizationScale=0.9,
                labels=fsm.topDownLabels,
                labelpos=1.1,
                labelskwargs=dict(fontsize=25),
            ),
        )
        fsm.HistoMaker_673(
            axes["Histo"],
            data[NPname],
            positions=[0, 1, 2, 9, 8, 3, 7, 5, 6, 4],
            barWidth=0.22,
            barSpace=0.03,
        )

        fig.savefig(f"673K/673_{NPname}.png", bbox_inches="tight", pad_inches=0, dpi=300)




    
# %%

# %%
import matplotlib.pyplot as plt
import h5py
from matplotlib.image import imread
import figureSupportModule as fsm
import numpy as np  # Importa numpy correttamente

np_values = ["01", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
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

plt.rcParams.update({"font.size": 25})


def addNPImages(axes, data, NP):
    clusters0K = []
    NClasses = 10
    for T in ["Ideal", 300]:
        p = [1.0]
        if T == "Ideal":
            ideal = data[T]["ClassTD"].references[0]
            clusters0K += [
                c for c in range(3, 10) if np.count_nonzero(ideal == c) > 0
            ]
        elif len(clusters0K) != 7:
            clusterCountNotID = np.zeros(
                (data[T]["ClassTD"].references.shape[0]), dtype=float
            )
            clusterCountID = np.zeros(
                (data[T]["ClassTD"].references.shape[0]), dtype=float
            )
            for c in range(3, 10):
                if c in clusters0K:
                    clusterCountID += np.count_nonzero(
                        data[T]["ClassTD"].references == c, axis=-1
                    )
            surfClusters = clusterCountID + clusterCountNotID
            clusterCountID /= surfClusters
            clusterCountNotID /= surfClusters
            p[0] = np.mean(clusterCountID)
            p += [np.mean(clusterCountNotID)]
        axes[f"np{T}"].imshow(imread(f"../nanop/{NP}_{T}-Topdown.png"))
        title = f"{T}\u2009K" if T != "Ideal" else "Starting Frame"
        plt.rcParams.update({"font.size": 20})
        axes[f"np{T}"].set_title(title)
        axes[f"np{T}"].axis("off")


for (NPname, t) in zip(np_values, trajs):
    with h5py.File(f"../../../{NPname}.hdf5", "r") as workFile:
        mask = workFile[f"Trajectories/{t}/Trajectory"][0, :, 2] > 84
    data = {}
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
    figsize = np.array([4, 7]) * 4
    fig, axes = fsm.makeLayoutSI_onlyT(
        labelsOptions=dict(fontsize=0), temps=[300], figsize=figsize, dpi=300
    )
    addNPImages(axes, data[NPname], NPname)

    for T in [300]:
        fsm.AddTmatsAndChord_onlyT(
            axes,
            data[NPname][T],
            T,
            zoom=0.020,
            cbarAx=None,  # if T != 673 else axes["tmatCMAP"],
            tmatOptions=dict(
                fmt="s",
                linewidth=1,
                # cbar_kws={} if T != 673 else {"label": "Probability"},
            ),
            chordOptions=dict(
                visualizationScale=0.9,
                labels=fsm.topDownLabels,
                labelpos=1.1,
                labelskwargs=dict(fontsize=25),
            ),
        )
        fsm.HistoMaker_300(
            axes["Histo"],
            data[NPname],
            positions=[0, 1, 2, 9, 8, 3, 7, 5, 6, 4],
            barWidth=0.22,
            barSpace=0.03,
        )

        fig.savefig(f"300K/300_{NPname}.png", bbox_inches="tight", pad_inches=0, dpi=300)

# %%
from PIL import Image
import os

# Cartella contenente le immagini 300K
image_folder_300K = "300K"
# Cartella contenente le immagini 673K
image_folder_673K = "673K"

# Lista dei nomi delle immagini (300K)
image_names_300K = [
    "300_01.png",
    "300_2.png",
    "300_3.png",
    "300_4.png",
    "300_5.png",
    "300_6.png",
    "300_7.png",
    "300_8.png",
    "300_9.png",
    "300_10.png",
]

# Lista dei nomi delle immagini (673K)
image_names_673K = [
    "673_01.png",
    "673_2.png",
    "673_3.png",
    "673_4.png",
    "673_5.png",
    "673_6.png",
    "673_7.png",
    "673_8.png",
    "673_9.png",
    "673_10.png",
]

# Funzione per unire le immagini e salvarle
def combine_and_save_images(image_folder, image_names, output_filename):
    # Apri e carica le immagini dalla cartella specificata
    images = [Image.open(os.path.join(image_folder, name)) for name in image_names]
    width, height = images[0].size

    # Calcola il numero di colonne necessarie
    num_columns = 5

    # Calcola il numero di righe necessarie
    num_rows = (len(images) + num_columns - 1) // num_columns  # 5 immagini per riga

    # Calcola le dimensioni dell'immagine totale
    total_width = width * num_columns
    total_height = height * num_rows

    # Crea un'immagine vuota con le dimensioni totali
    combined_image = Image.new("RGB", (total_width, total_height))

    # Unisci le immagini nelle righe e colonne
    for i, image in enumerate(images):
        column = i % num_columns
        row = i // num_columns
        combined_image.paste(image, (column * width, row * height))

    # Salva l'immagine totale
    combined_image.save(output_filename)

# Combina e salva le immagini dalla cartella 300K
combine_and_save_images(image_folder_300K, image_names_300K, "300K_combined_image.png")

# Combina e salva le immagini dalla cartella 673K
combine_and_save_images(image_folder_673K, image_names_673K, "673K_combined_image.png")


#%%

from figureSupportModule import (
    colorBarExporter,
    topDownColorMap,
    bottomUpColorMap,
    topDownFullColorMap,
)

#%%
colorBarExporter(topDownColorMap[::-1], "topDownCMAP.png")
#%%
#%%
colorBarExporter(topDownFullColorMap[::-1], "topDownFullCMAP.png")