#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 18:23:43 2021

@author: tibor
"""

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
import time
start_time = time.time()
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.join(currentdir, '..', '..', '..', 'config'))
sys.path.append(os.path.join(currentdir, '..', 'utilities'))
import config
from config import makeGlobalDM_TYPE
config.initialize()
from create_cat_dm import createCatDM
from create_cat_major_com_dm import createCatMajorCOMDM
from create_cat_major_com_gx import createCatMajorCOMGx

for snap in config.SNAP_ABB: # DM Catalogue generation
    
    makeGlobalDM_TYPE(snap, start_time)
    # DM Catalogue generation
    createCatDM()
    createCatMajorCOMDM()
    
    # Gxs Catalogue generation
    createCatDM()
    createCatMajorCOMGx()