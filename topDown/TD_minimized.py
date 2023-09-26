#%%
from h5py import File
from referenceMaker import getDefaultReferences, getDefaultReferencesSubdict
from SOAPify import getDistancesFromRefNormalized

refFile = "References.hdf5"
def calculatedDistancesAndSave(references, SOAPFileName, classificationFile):

    with File(
        SOAPFileName, "r" if SOAPFileName != classificationFile else "a"
    ) as workFile, File(classificationFile, "a") as distFile:
        g = workFile[f"SOAP"]
        distG = distFile.require_group("Distances")
        for key in g.keys():
            for refKey in references:
                t = getDistancesFromRefNormalized(g[key], references[refKey])
                dgd = distG.require_dataset(
                    f"{refKey}/{key}", shape=t.shape, dtype=t.dtype, chunks=True
                )
                dgd[:] = t
                dgd.attrs["Reference"] = f"{refFile}/{refKey}"
                dgd.attrs["names"] = references[refKey].names





if __name__ == "__main__":
    references = {k: getDefaultReferencesSubdict(k) for k in ["ih", "to", "dh"]}
    references["icotodh"] = getDefaultReferences()

    calculatedDistancesAndSave(references, "../minimized.hdf5", "../minimized.hdf5")
    
# %%
