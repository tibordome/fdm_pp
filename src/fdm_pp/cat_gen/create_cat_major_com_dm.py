#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 14:53:46 2021

@author: tibor
"""
import numpy as np
from morphology import getMorphologies
from get_hdf5 import getHDF5DMData
import json
from print_msg import print_status
import time
from copy import deepcopy
import h5py
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
import config

def createCatMajorCOMDM():          
    start_time = time.time()
    print_status(rank,start_time,'Starting createCatMajorCOMDM() with snap {0}'.format(config.SNAP))
    
    # Import hdf5 data and reading SH COMs
    print_status(rank,start_time,"Getting HDF5 raw data..")
    dm_xyz, dm_masses = getHDF5DMData()
    if rank == 0:
        with open('{0}/sh_cat_fdm_{1}.txt'.format(config.CAT_DEST, config.SNAP), 'r') as filehandle:
            cat = json.load(filehandle)
        rdelta = list(np.loadtxt('{0}/r_delta_fdm_{1}.txt'.format(config.CAT_DEST, config.SNAP)))
    else:
        cat = None
        rdelta = None
    cat = comm.bcast(cat, root = 0) 
    rdelta = comm.bcast(rdelta, root = 0) 
    print_status(rank, start_time, "Gotten HDF5 raw data. Number of FDM SHs is {0}".format(len(cat)))
    
    # Discard SHs with bad resolution
    print_status(rank, start_time, "Discarding SHs with bad resolution..")
    if rank == 0:
        sh_idx = 0
        while sh_idx < len(cat):
            if len(cat[sh_idx]) < config.MIN_NUMBER_DM_PTCS:
                cat = [x if idx != sh_idx else [] for idx, x in enumerate(cat)]  # Remove SH that has bad resolution
            else:
                sh_idx += 1
    cat = comm.bcast(cat, root = 0)
    print_status(rank, start_time, "The number of SHs before morphology calculation is {0}".format(len(cat)))
    
    # Retrieve 9*eps with updated cat
    if rank == 0:
        eps_9 = h5py.File(r'{0}/snap_{1}.{2}.hdf5'.format(config.HDF5_SNAP_DEST, config.SNAP, 0), 'r')['Parameters'].attrs['SofteningComovingType1']/1000*9 # In cMpc/h
        np.savetxt('{0}/eps_9_{1}_dm_{2}.txt'.format(config.CAT_DEST, config.DM_TYPE, config.SNAP), np.array([eps_9]), fmt='%1.7e')
        
    # Morphology
    print_status(rank, start_time, "Calculating morphologies. The average number of ptcs in the SHs is {0}".format(np.average(np.array(list(map(lambda x: len(cat[x]), range(len(cat))))))))
    d, q, s, minor, inter, major, shs_com, sh_m, succeeded = getMorphologies(dm_xyz, cat, dm_masses, rdelta, "dm", "local", start_time)
    print_status(rank, start_time, "Gotten morphologies")
    
    if rank == 0:
        if d != []:
            d = np.reshape(np.array(d), (np.array(d).shape[0], np.array(d).shape[1])) # Has shape (number_of_shs, d_discr), each d_discr a float
            q = np.reshape(np.array(q), (np.array(q).shape[0], np.array(q).shape[1])) # Has shape (number_of_shs, d_discr), each d_discr a float
            s = np.reshape(np.array(s), (np.array(s).shape[0], np.array(s).shape[1])) # Has shape (number_of_shs, d_discr), each d_discr a float
            minor = np.array(minor) # Has shape (number_of_shs, d_discr), each of (3,) shape
            inter = np.array(inter) # Has shape (number_of_shs, d_discr), each of (3,) shape
            major = np.array(major) # Has shape (number_of_shs, d_discr), each of (3,) shape
            major_new = np.zeros((major.shape[0], major.shape[1], 3)) # Has shape (number_of_shs, d_discr, 3) finally
            for sh_idx in range(major.shape[0]):
                for d_idx in range(major.shape[1]):
                    major_new[sh_idx, d_idx] = major[sh_idx, d_idx]
            inter_new = np.zeros((major.shape[0], major.shape[1], 3))
            for sh_idx in range(major.shape[0]):
                for d_idx in range(major.shape[1]):
                    inter_new[sh_idx, d_idx] = inter[sh_idx, d_idx]
            minor_new = np.zeros((major.shape[0], major.shape[1], 3))
            for sh_idx in range(major.shape[0]):
                for d_idx in range(major.shape[1]):
                    minor_new[sh_idx, d_idx] = minor[sh_idx, d_idx]
        
            minor_new_re = minor_new.reshape(minor_new.shape[0], -1)
            inter_new_re = inter_new.reshape(inter_new.shape[0], -1)
            major_new_re = major_new.reshape(major_new.shape[0], -1)
        
        else:
            d = np.array(d) # Empty array
            q = np.array(q) # Empty
            s = np.array(s) # Empty
            minor_new_re = np.array(minor) 
            inter_new_re = np.array(inter) 
            major_new_re = np.array(major) 
        shs_com = np.array(shs_com) # Has shape (number_of_shs, 3)
        sh_m = np.array(sh_m) # Has shape (number_of_shs, )
        
        # Create catalogue storing major axes at maximal r_ell and sh_com
        a_com_cat = [[] for i in range(d.shape[0])] # For each sh, 6 floats, first 3 give orientation of major axis at maximal r_ell, last 3 give sh_com's x, y, z
        
        for sh in range(d.shape[0]):
            a_com_cat[sh].extend((major[sh][-1][0], major[sh][-1][1], major[sh][-1][2], shs_com[sh][0], shs_com[sh][1], shs_com[sh][2], sh_m[sh]))
                
        # Writing
        with open('{0}/a_com_cat_local_fdm_dm_{1}.txt'.format(config.CAT_DEST, config.SNAP), 'w') as filehandle:
            json.dump(a_com_cat, filehandle)
           
        cat_local = deepcopy(cat)
        for sh_idx in range(len(cat_local)):
            if sh_idx not in succeeded: # We are removing those SHs whose rdelta shell does not converge.
                cat_local =  [x if idx != sh_idx else [] for idx, x in enumerate(cat_local)]
        with open('{0}/sh_cat_local_fdm_{1}.txt'.format(config.CAT_DEST, config.SNAP), 'w') as filehandle:
            json.dump(cat_local, filehandle)
        
        # Storing np.arrays
        np.savetxt('{0}/d_local_fdm_dm_{1}.txt'.format(config.CAT_DEST, config.SNAP), d, fmt='%1.7e')
        np.savetxt('{0}/q_local_fdm_dm_{1}.txt'.format(config.CAT_DEST, config.SNAP), q, fmt='%1.7e')
        np.savetxt('{0}/s_local_fdm_dm_{1}.txt'.format(config.CAT_DEST, config.SNAP), s, fmt='%1.7e')
        
        np.savetxt('{0}/minor_local_fdm_dm_{1}.txt'.format(config.CAT_DEST, config.SNAP), minor_new_re, fmt='%1.7e')
        np.savetxt('{0}/inter_local_fdm_dm_{1}.txt'.format(config.CAT_DEST, config.SNAP), inter_new_re, fmt='%1.7e')
        np.savetxt('{0}/major_local_fdm_dm_{1}.txt'.format(config.CAT_DEST, config.SNAP), major_new_re, fmt='%1.7e')