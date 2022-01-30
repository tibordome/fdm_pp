#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 10:09:58 2021

@author: tibor
"""

import numpy as np
import time
import json
import make_grid_nn
from get_hdf5 import getHDF5DMData
from print_msg import print_status
from fdm_utilities import getLocalMaxima3D, getMDelta, getPotential, getEnclosedMassProfiles, getDensityProfiles, constructCat
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
import config

def createCatDM():  
       
    start_time = time.time()
    print_status(rank,start_time,'Starting createCatDM() with config.Overdensity being {0}'.format(config.OVERDENSITY))
    
    # Import hdf5 data
    print_status(rank,start_time,"Getting HDF5 raw data..")
    dm_xyz, dm_masses = getHDF5DMData() 
    print_status(rank, start_time, "Gotten HDF5 raw data.")
                 
    # Construct rho grid
    if rank == 0:
        rho_tmp = make_grid_nn.makeGridWithNNPBC(dm_xyz[:,0].astype('float32'), dm_xyz[:,1].astype('float32'), dm_xyz[:,2].astype('float32'), dm_masses.astype('float32'), config.L_BOX, config.N).astype('float32') # Shape (config.N, config.N, config.N)
    else:
        rho_tmp = np.zeros((config.N,config.N,config.N), dtype = np.float32)
    rho_tmp = rho_tmp.flatten()
    nb_dm_ptcs = config.N**3
    pieces = 1 + (nb_dm_ptcs>=3*10**8)*nb_dm_ptcs//(3*10**8) # Not too high since this is a slow-down!
    chunk = nb_dm_ptcs//pieces
    rho = np.empty(0, dtype = np.float32)
    for i in range(pieces):
        to_bcast = rho_tmp[i*chunk:(i+1)*chunk+(i==(pieces-1))*(nb_dm_ptcs-pieces*chunk)]
        comm.Bcast(to_bcast, root=0)
        rho = np.hstack((rho, to_bcast))
    rho = np.reshape(rho, (config.N, config.N, config.N))
    rhobar = np.mean(rho)
    del rho_tmp
    print_status(rank, start_time, "Minimum density is {0} and maximum density is {1} while rhobar is {2}".format(rho.min(), rho.max(), rhobar))
    print_status(rank, start_time, "The maximum density is located at {0}".format(np.unravel_index(rho.argmax(), rho.shape)))

    # Find V
    print_status(rank, start_time, "Solve Poisson equation for gravitational potential V...")
    if rank == 0:
        V = getPotential(rho, rhobar, config.L_BOX, config.G) # Shape is (config.N, config.N, config.N)    
        print_status(rank, start_time, "Minimum V is {0} and maximum V is {1}".format(V.min(), V.max())) # Moved inside rank == 0 to avoid UnboundLocalError
        print_status(rank, start_time, "The minimum V is located at {0}".format(np.unravel_index(V.argmin(), V.shape)))

    # Finding local minima in V
    print_status(rank, start_time, "Find local maxima...")
    if rank == 0:
        coords, values = getLocalMaxima3D(-V, order = 1) # Find local minima in the potential field
        x_idx = coords[:,0]
        y_idx = coords[:,1]
        z_idx = coords[:,2]
        mask = rho[x_idx, y_idx, z_idx] > config.OVERDENSITY*rhobar # Discard peaks that are below Delta*rhobar threshold
        # Later we will also impose a stronger condition of 3**3 index nbh being all above Delta*rhobar
        coords = coords[mask].astype('float32')
        values = values[mask]
        Npeaks = coords.shape[0]
        del V
        print_status(rank, start_time, "The number of peaks is {0} and the first 5 peaks are located at {1} with -V values {2}".format(Npeaks, coords[:5], values[:5]))
    else:
        Npeaks = None
    Npeaks = comm.bcast(Npeaks, root = 0)
    if rank != 0:
        coords = np.zeros((Npeaks, 3), dtype = np.float32)
    comm.Bcast(coords, root = 0)
        
    # Calculate MDelta via enclosed mass profiles
    print_status(rank, start_time, "Determine enclosed mass profile...")
    rho_enc, R_enc, M_enc = getEnclosedMassProfiles(coords.astype('int32'), rho, Npeaks) # All have shapes [Npeaks, Nchar]
    print_status(rank, start_time, "Finished getEnclosedMassProfiles(), shapes are {0}, {1}, {2}".format(rho_enc.shape, R_enc.shape, M_enc.shape))
    print_status(rank, start_time, "The first peak has the following rho_enc, R_enc, and M_enc: {0}, {1}, {2}".format(rho_enc[0], R_enc[0], M_enc[0]))

    print_status(rank, start_time, "Determine MDelta values...")
    if rank == 0:
        M_Delta, R_Delta, invalids = getMDelta(rho_enc, rhobar, M_enc, R_enc, Npeaks) # All have shapes [Npeaks,]
        np.savetxt('{0}/m_delta_fdm_dm_{1}.txt'.format(config.CAT_DEST), M_Delta, config.SNAP, fmt='%1.7e')
        np.savetxt('{0}/r_delta_fdm_dm_{1}.txt'.format(config.CAT_DEST), R_Delta, config.SNAP, fmt='%1.7e')
        print_status(rank, start_time, "The first 5 M_Delta are {0}, the first 5 R_Delta are {1}, and the number of invalids is {2}".format(M_Delta[:5], R_Delta[:5], len(invalids)))
    else:
        R_Delta = np.zeros((Npeaks,), dtype = np.float32)
        invalids = None
    comm.Bcast(R_Delta, root = 0)
    invalids = comm.bcast(invalids, root = 0)
    
    # Density Profiles
    print_status(rank, start_time, "Get density profiles...")
    if rank == 0:
        rho_profiles = getDensityProfiles(coords, rho, Npeaks, invalids) # Shape [Npeaks, config.N]
        np.savetxt('{0}/rho_profiles_fdm_dm_{1}.txt'.format(config.CAT_DEST, config.SNAP), rho_profiles, fmt='%1.7e')
        print_status(rank, start_time, "Gotten density profiles...")
        print_status(rank, start_time, "The first peak has the following rho_profiles: {0}".format(rho_profiles[0]))

    # Construct catalogue
    print_status(rank, start_time, "Constructing catalogue...")
    coms = coords*config.DEL_X # COMs have to be on grid points
    sh_cat = constructCat(dm_xyz.astype('float32'), coms.astype('float32'), R_Delta.astype('float32'), Npeaks)
    print_status(rank, start_time, "Gotten catalogue")
    
    # Writing
    if rank == 0:
        with open('{0}/sh_cat_fdm_{1}.txt'.format(config.CAT_DEST, config.SNAP), 'w') as filehandle:
            json.dump(sh_cat, filehandle)
            
        np.savetxt('{0}/sh_coms_fdm_{1}.txt'.format(config.CAT_DEST, config.SNAP), coms, fmt='%1.7e')