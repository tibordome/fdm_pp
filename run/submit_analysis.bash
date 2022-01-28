#!/bin/bash

cd ../output
bash create_dir.bash # Won't be harmful if they already exist
cd ../src/shape_al_fdm/analysis
rm make_grid_cic.c
rm make_grid_cic.so
python3 setup.py build_ext --inplace
mpirun -n 7 python3 run_analysis.py