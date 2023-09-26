#!/bin/bash

source ../NPenv/bin/activate
#source ~/telemessage.sh

#python TDcomplete.py
echo "#hdf5: TDComplete  exited with code $?"
python TD_minimized.py
python TD_01calculateDistancesFromReferences.py
python TD02_elaborateDistances.py


