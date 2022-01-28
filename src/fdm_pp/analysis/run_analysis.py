#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 18:23:43 2021

@author: tibor
"""
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.join(currentdir, '..', '..', '..', 'config'))
sys.path.append(os.path.join(currentdir, '..', 'utilities'))
from print_msg import print_status
from analyze_gx_alignments import analyze_gx_alignments
from analyze_gx_shapes import analyze_gx_shapes
from analyze_dm_alignments import analyze_dm_alignments
from dm_shapes import dm_shapes
from dm_profiles import dm_profiles
from majors_projected_dm import projectMajorsHalo
from majors_projected_gx import projectMajorsGx
import config
from config import makeGlobalHALO_REGION, makeGlobalDM_TYPE
config.initialize()
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
import time
start_time = time.time()


print_status(rank, start_time, "Running shape/alignment analysis script with N = {0}, L = {1}, D_BINS = {2}, HALO_REGION = {3}, M_SPLIT_TYPE = {4}, CUT_LEFT = {5}, CUT_RIGHT = {6}, SNAP_ABB = {7}".format(config.N, config.L_BOX, config.D_BINS, config.HALO_REGION, config.M_SPLIT_TYPE, config.CUT_LEFT, config.CUT_RIGHT, config.SNAP_ABB))

for snap in config.SNAP_ABB:
    
    # NEXUS analysis
    #vol_mass_frac(start_time) # Changes snap, but reversed in next line
    
    makeGlobalDM_TYPE(snap, start_time)
    
    # Gx alignment analysis
    #analyze_gx_alignments()
    #analyze_gx_shapes()
    
    # SH alignment analysis
    makeGlobalHALO_REGION('Full')
    dm_shapes(start_time) # For ellipticity histogram only
    makeGlobalHALO_REGION('Inner')
    dm_shapes(start_time)
    dm_profiles(start_time)
    
    # Projection figures
    projectMajorsHalo(start_time)
    projectMajorsGx(start_time)