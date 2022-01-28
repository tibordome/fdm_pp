#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 14:53:46 2021

@author: tibor
"""

import numpy as np
from morphology import getMorphologies
import json
from get_hdf5 import getHDF5StarData
import time
from print_msg import print_status
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
import config

def createCatMajorCOMGx():
    start_time = time.time()
    print_status(rank,start_time,'Starting createCatMajorCOMGx()')
    
    # Import hdf5 data and reading SH COMs
    print_status(rank,start_time,"Getting HDF5 raw data..")
    star_xyz, star_masses = getHDF5StarData()     
    sh_com = np.loadtxt('{0}/sh_coms_fdm.txt'.format(config.CAT_DEST))
    print_status(rank, start_time, "Gotten HDF5 raw data. Number of SHs is {0}".format(sh_com.shape[0]))

    # Defining galaxies: Method 1: 1 csh = at most 1 galaxy
    print_status(rank, start_time, "Creating Gx CAT..")
    perrank = star_xyz.shape[0]//size
    comm.Barrier()
    gx_cat = [[] for i in range(sh_com.shape[0])]
    last = rank == size - 1 # Whether or not last process
    for star_ptc in range(rank*perrank, (rank+1)*perrank+last*(star_xyz.shape[0]-(rank+1)*perrank)):
        dist_x = abs(star_xyz[star_ptc,0]-sh_com[:,0])
        dist_x[dist_x > config.L_BOX/2] = config.L_BOX-dist_x[dist_x > config.L_BOX/2]
        dist_y = abs(star_xyz[star_ptc,1]-sh_com[:,1])
        dist_y[dist_y > config.L_BOX/2] = config.L_BOX-dist_y[dist_y > config.L_BOX/2]
        dist_z = abs(star_xyz[star_ptc,2]-sh_com[:,2])
        dist_z[dist_z > config.L_BOX/2] = config.L_BOX-dist_z[dist_z > config.L_BOX/2]
        argmin = np.argmin(dist_x**2+dist_y**2+dist_z**2)
        gx_cat[argmin].append(star_ptc) # In case star_ptc is exactly equally close to multiple subhalos, argmin will be first subhalo
    gx_cat = comm.gather(gx_cat, root = 0)
    if rank == 0:
        gx_cat_full = [[] for i in range(sh_com.shape[0])]
        for r in range(size):
            for sh in range(sh_com.shape[0]):
                gx_cat_full[sh] += gx_cat[r][sh]
    else:
        gx_cat_full = None
    gx_cat = comm.bcast(gx_cat_full, root = 0)
    print_status(rank, start_time, "Gotten Gx CAT")

    # Discard gxs with bad resolution
    if rank == 0:
        gx_idx = 0
        while gx_idx < len(gx_cat):
            if len(gx_cat[gx_idx]) < config.MIN_NUMBER_STAR_PTCS:
                gx_cat = [x for idx, x in enumerate(gx_cat) if idx != gx_idx] # Remove gx that has bad resolution
            else:
                gx_idx += 1
    gx_cat = comm.bcast(gx_cat, root = 0)
    print_status(rank, start_time, "The number of gxs before morphology calculation is {0}".format(len(gx_cat)))
    
    # Morphology
    print_status(rank, start_time, "Calculating morphologies. The average number of ptcs in the gxs is {0}".format(np.average(np.array(list(map(lambda x: len(gx_cat[x]), range(len(gx_cat))))))))
    d, q, s, minor, inter, major, gx_com, gx_m = getMorphologies(star_xyz, gx_cat, star_masses, "gxs", start_time)
    print_status(rank, start_time, "Gotten morphologies")
    
    if rank == 0:
        d = np.reshape(np.array(d), (np.array(d).shape[0], np.array(d).shape[1])) # Has shape (number_of_gxs, d_discr), each d_discr a float
        q = np.reshape(np.array(q), (np.array(q).shape[0], np.array(q).shape[1])) # Has shape (number_of_gxs, d_discr), each d_discr a float
        s = np.reshape(np.array(s), (np.array(s).shape[0], np.array(s).shape[1])) # Has shape (number_of_gxs, d_discr), each d_discr a float
        minor = np.array(minor) # Has shape (number_of_gxs, d_discr), each of (3,) shape
        inter = np.array(inter) # Has shape (number_of_gxs, d_discr), each of (3,) shape
        major = np.array(major) # Has shape (number_of_gxs, d_discr), each of (3,) shape
        gx_com = np.array(gx_com) # Has shape (number_of_gxs, 3)
        gx_m = np.array(gx_m) # Has shape (number_of_gxs, )
        
        # Create catalogue storing major axes at maximal r_ell and gx_com
        a_com_cat_gx = [[] for i in range(d.shape[0])] # For each gx, 6 floats, first 3 give orientation of major axis at maximal r_ell, last 3 give gx_com's x, y, z
        
        for gx in range(d.shape[0]):
            a_com_cat_gx[gx].extend((major[gx][-1][0], major[gx][-1][1], major[gx][-1][2], gx_com[gx][0], gx_com[gx][1], gx_com[gx][2], gx_m[gx]))
        
        # Writing
        with open('{0}/a_com_cat_fdm_gx.txt'.format(config.CAT_DEST), 'w') as filehandle:
            json.dump(a_com_cat_gx, filehandle)
       
        # Storing np.arrays
        np.savetxt('{0}/d_fdm_gx.txt'.format(config.CAT_DEST), d, fmt='%1.7e')
        np.savetxt('{0}/q_fdm_gx.txt'.format(config.CAT_DEST), q, fmt='%1.7e')
        np.savetxt('{0}/s_fdm_gx.txt'.format(config.CAT_DEST), s, fmt='%1.7e')
        
        major_new = np.zeros((major.shape[0], major.shape[1], 3)) # Has shape (number_of_gxs, d_discr, 3) finally
        for gx in range(major.shape[0]):
            for d_idx in range(major.shape[1]):
                major_new[gx, d_idx] = major[gx, d_idx]
        inter_new = np.zeros((major.shape[0], major.shape[1], 3))
        for gx in range(major.shape[0]):
            for d_idx in range(major.shape[1]):
                inter_new[gx, d_idx] = inter[gx, d_idx]
        minor_new = np.zeros((major.shape[0], major.shape[1], 3))
        for gx in range(major.shape[0]):
            for d_idx in range(major.shape[1]):
                minor_new[gx, d_idx] = minor[gx, d_idx]
        
        minor_new_re = minor_new.reshape(minor_new.shape[0], -1)
        inter_new_re = inter_new.reshape(inter_new.shape[0], -1)
        major_new_re = major_new.reshape(major_new.shape[0], -1)
        np.savetxt('{0}/minor_fdm_gx.txt'.format(config.CAT_DEST), minor_new_re, fmt='%1.7e')
        np.savetxt('{0}/inter_fdm_gx.txt'.format(config.CAT_DEST), inter_new_re, fmt='%1.7e')
        np.savetxt('{0}/major_fdm_gx.txt'.format(config.CAT_DEST), major_new_re, fmt='%1.7e')