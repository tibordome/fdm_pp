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
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
import config

def createCatMajorCOMDM():          
    start_time = time.time()
    print_status(rank,start_time,'Starting createCatMajorCOMDM()')
    
    # Import hdf5 data and reading SH COMs
    print_status(rank,start_time,"Getting HDF5 raw data..")
    dm_xyz, dm_masses = getHDF5DMData()
    if rank == 0:
        with open('{0}/sh_cat_fdm.txt'.format(config.CAT_DEST), 'r') as filehandle:
            cat = json.load(filehandle)
    else:
        cat = None
    cat = comm.bcast(cat, root = 0) 
    print_status(rank, start_time, "Gotten HDF5 raw data. Number of FDM CSHs is {0}".format(len(cat)))
    
    # Discard CSHs with bad resolution
    print_status(rank, start_time, "Discarding CSHs with bad resolution..")
    if rank == 0:
        csh_idx = 0
        while csh_idx < len(cat):
            if len(cat[csh_idx]) < config.MIN_NUMBER_DM_PTCS:
                cat = [x for idx, x in enumerate(cat) if idx != csh_idx] # Remove CSH that has bad resolution
            else:
                csh_idx += 1
    cat = comm.bcast(cat, root = 0)
    print_status(rank, start_time, "The number of CSHs before morphology calculation is {0}".format(len(cat)))
        
    # Morphology
    print_status(rank, start_time, "Calculating morphologies. The average number of ptcs in the CSHs is {0}".format(np.average(np.array(list(map(lambda x: len(cat[x]), range(len(cat))))))))
    d, q, s, minor, inter, major, shs_com, sh_m = getMorphologies(dm_xyz, cat, dm_masses, "dm", start_time)
    print_status(rank, start_time, "Gotten morphologies")
    
    if rank == 0:
        d = np.reshape(np.array(d), (np.array(d).shape[0], np.array(d).shape[1])) # Has shape (number_of_shs, d_discr), each d_discr a float
        q = np.reshape(np.array(q), (np.array(q).shape[0], np.array(q).shape[1])) # Has shape (number_of_shs, d_discr), each d_discr a float
        s = np.reshape(np.array(s), (np.array(s).shape[0], np.array(s).shape[1])) # Has shape (number_of_shs, d_discr), each d_discr a float
        minor = np.array(minor) # Has shape (number_of_shs, d_discr), each of (3,) shape
        inter = np.array(inter) # Has shape (number_of_shs, d_discr), each of (3,) shape
        major = np.array(major) # Has shape (number_of_shs, d_discr), each of (3,) shape
        shs_com = np.array(shs_com) # Has shape (number_of_shs, 3)
        sh_m = np.array(sh_m) # Has shape (number_of_shs, )
        
        # Create catalogue storing major axes at maximal r_ell and sh_com
        a_com_cat = [[] for i in range(d.shape[0])] # For each sh, 6 floats, first 3 give orientation of major axis at maximal r_ell, last 3 give sh_com's x, y, z
        
        for sh in range(d.shape[0]):
            a_com_cat[sh].extend((major[sh][-1][0], major[sh][-1][1], major[sh][-1][2], shs_com[sh][0], shs_com[sh][1], shs_com[sh][2], sh_m[sh]))
                
        # Writing
        with open('{0}/a_com_cat_fdm_dm.txt'.format(config.CAT_DEST), 'w') as filehandle:
            json.dump(a_com_cat, filehandle)
           
        # Storing np.arrays
        np.savetxt('{0}/d_fdm_dm.txt'.format(config.CAT_DEST), d, fmt='%1.7e')
        np.savetxt('{0}/q_fdm_dm.txt'.format(config.CAT_DEST), q, fmt='%1.7e')
        np.savetxt('{0}/s_fdm_dm.txt'.format(config.CAT_DEST), s, fmt='%1.7e')
        
        major_new = np.zeros((major.shape[0], major.shape[1], 3)) # Has shape (number_of_shs, d_discr, 3) finally
        for sh in range(major.shape[0]):
            for d_idx in range(major.shape[1]):
                major_new[sh, d_idx] = major[sh, d_idx]
        inter_new = np.zeros((major.shape[0], major.shape[1], 3))
        for sh in range(major.shape[0]):
            for d_idx in range(major.shape[1]):
                inter_new[sh, d_idx] = inter[sh, d_idx]
        minor_new = np.zeros((major.shape[0], major.shape[1], 3))
        for sh in range(major.shape[0]):
            for d_idx in range(major.shape[1]):
                minor_new[sh, d_idx] = minor[sh, d_idx]
        
        minor_new_re = minor_new.reshape(minor_new.shape[0], -1)
        inter_new_re = inter_new.reshape(inter_new.shape[0], -1)
        major_new_re = major_new.reshape(major_new.shape[0], -1)
        np.savetxt('{0}/minor_fdm_dm.txt'.format(config.CAT_DEST), minor_new_re, fmt='%1.7e')
        np.savetxt('{0}/inter_fdm_dm.txt'.format(config.CAT_DEST), inter_new_re, fmt='%1.7e')
        np.savetxt('{0}/major_fdm_dm.txt'.format(config.CAT_DEST), major_new_re, fmt='%1.7e')