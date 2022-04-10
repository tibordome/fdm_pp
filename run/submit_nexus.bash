#!/bin/bash

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/vault/td448/DisPerSE/external/gmp/gmp-4.2.4/install/usr/local/lib # only for HTCondor

cd ../output
bash create_dir.bash
cd ../src/shape_al/utilities
rm make_grid_cic.c
rm make_grid_cic.so
rm make_grid_sph.c
rm make_grid_sph.so
python3 setup.py build_ext --inplace
cd ../nexus
rm nexus_plus.cpp
rm nexus_plus.so
python3 setup.py build_ext --inplace
python3 run_nexus_plus.py
