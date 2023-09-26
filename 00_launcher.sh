#!/bin/bash

source ~/Scrivania/AuNanop/Analisi/RepoForGold/NPenv/bin/activate
#source ~/telemessage.sh

#python createHDF5ShortArgs.py ../trajectories/frame_*.lammpsdump
echo "#hdf5e5:  exited with code $?"
for h5 in *.hdf5; do
    if [[ "$h5"  == *soap* ]]; then
        continue
    fi
    #python SoapifyArgs.py "$h5"
    echo "#SOAP: ${h5} exited with code $?"
done

python createHDF5Minimized.py *.data
