#%%
from HDF5er import getXYZfromTrajGroup
import h5py
import figureSupportModule as fsm
from SOAPify import SOAPclassification

#%% Trajectories
for NPname in ["01","2","3","4","5","6","8","9","10"]:
#for NPname in ["3","4","5","6","8","9"]:
    print(f"{NPname}:")
    data = {673: {}, 300:{}}
    # loading data
    #fsm.loadClassificationBottomUp(f"../{NPname}classifications.hdf5", data , NPname)
    #fsm.dataLoaderTopDown(f"../{NPname}TopBottom.hdf5", data, NPname)
    fsm.loadClassificationTopDown(f"../{NPname}TopBottom.hdf5", data, NPname)
    # Exporting xyz
    with h5py.File(f"../{NPname}.hdf5", "r") as trajFile:
        g = trajFile["Trajectories"]
        for k in g:
            T = fsm.getT(k)
            print(f"\t{T}")
            if T in data:
                nat = g[k]["Types"].shape[-1]
                with open(f"{NPname}_{T}lastUs@1ns.xyz", "w") as icoFile:
                    getXYZfromTrajGroup(
                         icoFile,
                        g[k],
                        allFramesProperty='Origin="-0 -0 -0"',
                        topDown=data[T]["ClassTD"].references,
                        #bottomUP=data[T]["labelsNN"].reshape(-1, nat),
                    )
#%% Ideals:
for NPname in ["01","2","3","4","5","6","8","9","10"]:
#for NPname in ["1"]:
#    print(f"{NPname}:")
    data = {"Ideal": {}}
    T = "Ideal"
    # loading data
    with h5py.File(f"../minimized.hdf5", "r") as f:
        Simulations = f["/Classifications/icotodh"]
        #data["Ideal"]["labelsNN"] = Simulations[NPname]["labelsNN"][:].reshape(-1)
        #fsm.loadClassificationBottomUp(data, f"../minimized.hdf5")
        fsm.loadClassificationTopDown("../minimized.hdf5", data, NPname)
    # Exporting xyz
    with h5py.File(f"../minimized.hdf5", "r") as trajFile:
        g = trajFile[f"Trajectories/{NPname}"]
        with open(f"{NPname}_{T}.xyz", "w") as icoFile:
            getXYZfromTrajGroup(
                icoFile,
                g,
                allFramesProperty='Origin="0 0 0"',
                topDown=data[T]["ClassTD"].references,
                #bottomUP=data[T]["labelsNN"].reshape(-1, nat),
            )

# %%
