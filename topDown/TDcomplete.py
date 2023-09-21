from referenceMaker import getDefaultReferences, getDefaultReferencesSubdict
from TD_01calculateDistancesFromReferences import calculatedDistancesAndSave
from TD02_elaborateDistances import elaborateDistancesAndSave

if __name__ == "__main__":
    references = {k: getDefaultReferencesSubdict(k) for k in ["ih", "to", "dh"]}
    references["icotodh"] = getDefaultReferences()

    refFile = "References.hdf5"
    for NPname in ["01","2","3","4","5","6","7","8","9","10"]:
        SOAPFileName = f"../{NPname}soap.hdf5"
        classificationFile = f"../{NPname}TopBottom.hdf5"
        calculatedDistancesAndSave(references, SOAPFileName, classificationFile)
        elaborateDistancesAndSave(classificationFile)
