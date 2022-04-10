#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 11:47:39 2021

@author: tibor
"""

import h5py
import numpy as np
from astropy.io import fits as pyfits
import os
import sys
import subprocess
import argparse
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
subprocess.call(['python3', 'setup.py', 'build_ext', '--inplace'], cwd=os.path.join(currentdir, '..', 'analysis'))
sys.path.append(os.path.join(currentdir, '..', 'analysis'))
import make_grid_cic
sys.path.append(os.path.join(currentdir, '..', '..', '..', 'config'))
import time
start_time = time.time()
import config
from config import makeGlobalDM_TYPE
config.initialize()

# Argparsing
parser = argparse.ArgumentParser()
parser.add_argument('-l', '--list', help='delimited list input', type=str)
args = parser.parse_args()
my_list = [item for item in args.list.split(' ')]
for entry in my_list:
    if entry[0] == "0":
        snap = entry
    else:
        dm_type = entry
makeGlobalDM_TYPE(dm_type, snap, start_time)

# From HDF5 files
x_vec = np.empty(0, dtype = np.float32)
y_vec = np.empty(0, dtype = np.float32)
z_vec = np.empty(0, dtype = np.float32)
for i in range(config.SNAP_MAX):
    f = h5py.File(r'{0}/snap_{1}.{2}.hdf5'.format(config.HDF5_SNAP_DEST, snap, i), 'r')
    x_vec = np.hstack((x_vec, np.float32(f['PartType1/Coordinates'][:,0]/1000))) # in Mpc = 3.085678e+27 cm
    y_vec = np.hstack((y_vec, np.float32(f['PartType1/Coordinates'][:,1]/1000))) 
    z_vec = np.hstack((z_vec, np.float32(f['PartType1/Coordinates'][:,2]/1000)))
dm_masses = np.ones((x_vec.shape[0],), dtype=np.float32)*np.float32(f['Header'].attrs['MassTable'][1]) # in 1.989e+43 g
print("Working on snap {0}. Number of points is {1}.".format(snap, x_vec.size))

# Paint onto grid with config.FAST_DISPERSE ckpc/h spacing using CIC
grid = make_grid_cic.makeGridWithCICPBC(x_vec, y_vec, z_vec, dm_masses, config.L_BOX, int(config.L_BOX/config.FAST_DISPERSE))
print("Minimum density is {0} and maximum density is {1} while rhobar is {2}.".format(grid.min(), grid.max(), np.mean(grid)))

# Store grid as FITS file
outfile = 'DM_{0}_{1}.fits'.format(dm_type, snap)
pyfits.writeto(outfile, grid, overwrite=True)