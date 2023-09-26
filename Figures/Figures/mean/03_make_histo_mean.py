#%%
# This script generates and displays histograms for the 10 datasets at 673K

import matplotlib.pyplot as plt
import h5py
from matplotlib.image import imread
import figureSupportModule as fsm
import os
import numpy as np

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
    # for NPname in ["10"]:
    data[NPname] = dict()
    classificationFile = f"../../../{NPname}TopBottom.hdf5"
    fsm.loadClassificationTopDown(
        classificationFile, data[NPname], NPname, atomMask=mask
    )
    fsm.loadClassificationTopDown("../../../minimized.hdf5", data[NPname], NPname, atomMask=mask)
    # figsize = numpy.array([3.8, 3]) * 4

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.rcParams.update({"font.size": 21})
    ax = fsm.HistoMaker673(
        ax,
        data[NPname],
        positions=[0, 1, 2, 9, 8, 3, 7, 5, 6, 4],
        barWidth=0.55,  # sia ideal che 673
        barSpace=0.01,
        # barWidth=0.45,
        # barSpace=0.01,
    )
    plt.show()
    histo673 = data[NPname][673]["ClassTD"].references
    #numpy.savetxt(f"histo673_{NPname}", histo673)

    
# %%
# This script generates and displays histograms for the 10 datasets at 300K

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
    # figsize = numpy.array([3.8, 3]) * 4

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.rcParams.update({"font.size": 21})
    ax = fsm.HistoMaker300(
        ax,
        data[NPname],
        positions=[0, 1, 2, 9, 8, 3, 7, 5, 6, 4],
        barWidth=0.55,  # sia ideal che 673
        barSpace=0.01,
        # barWidth=0.45,
        # barSpace=0.01,
    )
    plt.show()
    histo673 = data[NPname][673]
    histo300 = data[NPname][300]



# %%
def makemean(data: dict):
    order = list(data.keys())  # Utilizza tutte le chiavi presenti in data
    nHisto = len(data[order[0]]["ClassTD"].references)
    countMean = {T: numpy.zeros((nHisto), dtype=float) for T in order}
    countDev = {T: numpy.zeros((nHisto), dtype=float) for T in order}
    pos = {T: numpy.zeros((nHisto), dtype=float) for T in order}
    for c in range(nHisto):
        for T in order:
            if T == "Ideal":
                continue
            countC = numpy.count_nonzero(data[T]["ClassTD"].references == c, axis=-1)
            countMean[T][c] = numpy.mean(countC)
            countDev[T][c] = numpy.std(countC)
    return countMean, countDev, pos


#%%


# Define a list of NP names and trajectories
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
all_results = {}  # Dictionary to store all results

# Iterate through NP names and calculate results
for (NPname, t) in zip(np, trajs):
    with h5py.File(f"../../../{NPname}.hdf5", "r") as workFile:
        mask = workFile[f"Trajectories/{t}/Trajectory"][0, :, 2] > 84
    data = {}
    data[NPname] = dict()
    classificationFile = f"../../../{NPname}TopBottom.hdf5"
    
    # Load classification data
    fsm.loadClassificationTopDown(
        classificationFile, data[NPname], NPname, atomMask=mask
    )
    
    # Calculate mean results
    results = makemean(data[NPname])
    all_results[NPname] = results 

# Write countMean values to separate text files for each NP without zeros
for NPname, results in all_results.items():
    countMean, _, _ = results
    file_name = f"{NPname}_countMean_values.txt"
    
    with open(file_name, "w") as file:
        file.write(f"CountMean values for {NPname}:\n")
        
        # Iterate through keys and values, formatting and excluding zeros
        for key, value in countMean.items():
            formatted_values = [f"{val:.2f}" for val in value if val != 0]  # Format values with two decimal places and exclude zeros
            if formatted_values:  # Check if there are non-zero values to write
                file.write(f"{key}: {', '.join(formatted_values)}\n")


# %%



# List of NP names
np_names = ["01", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

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
all_results = {}

# Iterate through NP names and calculate results
for (NPname, t) in zip(np_names, trajs):
    with h5py.File(f"../../../{NPname}.hdf5", "r") as workFile:
        if f"Trajectories/{t}/Trajectory" in workFile:
            mask = workFile[f"Trajectories/{t}/Trajectory"][0, :, 2] > 84
        else:
            mask = None
    data = {}
    data[NPname] = dict()
    classificationFile = f"../../../{NPname}TopBottom.hdf5"
    fsm.loadClassificationTopDown(
        classificationFile, data[NPname], NPname, atomMask=mask
    )
    results = makemean(data[NPname])
    all_results[NPname] = results

# Write the values of countMean to separate files for each NP and T-value
for NPname, results in all_results.items():
    countMean, _, _ = results
    # Create a dictionary to group values by NP and T-value
    grouped_values = {}
    for key, value in countMean.items():
        if isinstance(key, str):
            T_value = key.split(":")[0]
            if T_value not in grouped_values:
                grouped_values[T_value] = {}
            # Exclude zeros and remove the "T" prefix from names
            filtered_values = {k: v for k, v in value.items() if v != 0 and not k.startswith('T')}
            grouped_values[T_value].update(filtered_values)

    # Write the values grouped by T-value to the appropriate files
    for T_value, values in grouped_values.items():
        file_name = f"{NPname}_countMean_values.txt"
        # Check if the file exists before opening it
        if values:
            with open(file_name, "w") as file:
                formatted_values = [f"{k}: {v:.2f}" for k, v in values.items()]  # Format values with two decimal places
                file.write(", ".join(formatted_values))

# Split the files into two, one for "300" and one for "673"
for NPname in np_names:
    for T_value in ["300", "673"]:
        input_file_name = f"{NPname}_countMean_values.txt"
        output_file_name = f"{T_value}_{NPname}_histo.txt"

        # Check if the input file exists before opening it
        if os.path.exists(input_file_name):
            with open(input_file_name, "r") as input_file, open(output_file_name, "w") as output_file:
                lines = input_file.readlines()
                for line in lines:
                    if line.strip().startswith(T_value + ":"):
                        data_values = line.strip().split(":")[1].strip()  # Extract data after the colon
                        output_file.write(data_values + "\n")  # Write the data to the new file


# %%
import numpy as np
np_names = ["01", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

# Define the desired T_values
T_values = ["300", "673"]

# Iterate through the T_values
for T_value in T_values:
    # Create an empty list to store the averages of the first values from the 10 files
    averages = [0] * 10
    
    # Iterate through the NP names
    for NPname_index, NPname in enumerate(np_names):
        file_name = f"{T_value}_{NPname}_histo.txt"
        
        # Check if the file exists before opening it
        if os.path.exists(file_name):
            with open(file_name, "r") as file:
                lines = file.readlines()
                if lines:
                    # Read values from the file (presumably there is one value per line)
                    values_str = lines[0].strip().split(', ')
                    values_float = [float(val) for val in values_str]
                    if len(values_float) == 10:
                        averages[NPname_index] = values_float
    
    # Calculate the average of the first values from all files
    # The zip(*averages) function transposes the list of lists
    average_values = [np.mean(values) for values in zip(*averages)]
    
    # Write the averages to the output file
    output_file_name = f"{T_value}_average_histo.txt"
    with open(output_file_name, "w") as output_file:
        formatted_values = [f"{average:.2f}" for average in average_values]
        output_file.write(", ".join(formatted_values))



#%%

# %%
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# Define the input file names for 300 K and 673 K
filename_300 = "300_average_histo.txt"
filename_673 = "673_average_histo.txt"

# Initialize empty lists to store the data
average300 = []
average673 = []

# Read the data from the input files
with open(filename_300, "r") as file_300:
    for line in file_300:
        values = line.strip().split(',')
        average300.extend([float(value) for value in values])

with open(filename_673, "r") as file_673:
    for line in file_673:
        values = line.strip().split(',')
        average673.extend([float(value) for value in values])

def HistoMakermean(
    ax: plt.Axes,
    data: list,
    barWidth: float = 0.15,
    barSpace: float = 0.05,
    arrowLength: float = 0.25,
    positions=None,
    temperature=None,
):

    nHisto = len(data)

    if positions is None:
        positions = range(nHisto)

    order = [temperature]

    pos = {T: np.zeros((nHisto), dtype=float) for T in order}

    barsDim = 1 * barWidth + 10 * barSpace

    for c in range(nHisto):
        dev = -barsDim / 13
        for T in order:
            pos[T][c] = positions[c] + dev
            dev += barWidth + barSpace

    styles = {
        300: dict(hatch="\\\\\\", edgecolor="w"),
        673: dict(hatch="///", edgecolor="w"),
    }

    for i, T in enumerate(order):
        ax.bar(
            pos[T],
            data,
            width=barWidth,
            align="center",
            color=fsm.topDownColorMap,  # Use the specified color
            **styles[T],
        )

    scaleAx = 0.6
    ax.set_xticks(
        range(nHisto),
        [fsm.topDownLabels[p] for p in positions],
        fontsize=25,
    )

    ax.set_xlim(-barsDim * scaleAx, nHisto - 1 + barsDim * scaleAx)
    ax.set_ylabel("Mean Number of Atoms")
    
    # Set the title to indicate the temperature
    ax.set_title(f" Averaged histograms for  {temperature} K")
    
    # Optionally, you can add a legend for the bars
    legend_elements = [Patch(label=f"{temperature} K", hatch=styles[temperature]["hatch"], edgecolor=styles[temperature]["edgecolor"])]
    ax.legend(handles=legend_elements, fontsize=12)

# Create a figure and axis for the plots
fig, ax = plt.subplots(2, 1, figsize=(10, 16))
plt.rcParams.update({"font.size": 12})

# Call the HistoMakermean function to plot the data for 300 K
ax[0] = HistoMakermean(
    ax[0],
    average300,
    positions=[0, 1, 2, 9, 8, 3, 7, 5, 6, 4],
    barWidth=0.55,
    barSpace=0.01,
    temperature=300,  # Specify the temperature
)

# Call the HistoMakermean function to plot the data for 673 K
ax[1] = HistoMakermean(
    ax[1],
    average673,
    positions=[0, 1, 2, 9, 8, 3, 7, 5, 6, 4],
    barWidth=0.55,
    barSpace=0.01,
    temperature=673,  # Specify the temperature
)

# Show the plots
plt.show()



# %%
