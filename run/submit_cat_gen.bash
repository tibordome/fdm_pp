#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/vault/td448/DisPerSE/external/gmp/gmp-4.2.4/install/usr/local/lib # only for HTCondor

cd ../output
bash create_dir.bash
cd ../src/shape_al_fdm/cat_gen
rm make_grid_nn.c
rm make_grid_nn.so
python3 setup.py build_ext --inplace
mpirun -n 8 python3 create_catalogue.py
