#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/vault/td448/DisPerSE/external/gmp/gmp-4.2.4/install/usr/local/lib # only for HTCondor

cd ../output
bash create_dir.bash
cd ../src/fdm_pp/utilities
rm make_grid_cic.c
rm make_grid_cic.so
rm make_grid_nn.c
rm make_grid_nn.so
rm fdm_utilities.cpp
rm fdm_utilities.so
python3 setup.py build_ext --inplace
cd ../analysis
mpirun -n 7 python3 run_analysis.py
