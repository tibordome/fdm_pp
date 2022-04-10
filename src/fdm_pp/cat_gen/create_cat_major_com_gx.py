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
import h5py
from copy import deepcopy
from print_msg import print_status
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
import config

def createCatMajorCOMGx(start_time):
    print_status(rank,start_time,'Starting createCatMajorCOMGx() with snap {0}'.format(config.SNAP))
    
    # Import hdf5 data and reading SH COMs
    print_status(rank,start_time,"Getting HDF5 raw data..")
    star_xyz, star_masses = getHDF5StarData()     
    if rank == 0:
        with open('{0}/sh_cat_fdm_{1}.txt'.format(config.CAT_DEST, config.SNAP), 'r') as filehandle:
            sh_cat = json.load(filehandle)
        rdelta = list(np.loadtxt('{0}/r_delta_fdm_{1}.txt'.format(config.CAT_DEST, config.SNAP)))
    else:
        sh_cat = None
        rdelta = None
    sh_cat = comm.bcast(sh_cat, root = 0)
    rdelta = comm.bcast(rdelta, root = 0)
    if rank == 0:
        sh_com = np.float32(np.loadtxt('{0}/sh_coms_fdm_{1}.txt'.format(config.CAT_DEST, config.SNAP)))
    else:
        sh_com = np.zeros((len(rdelta), 3), dtype = np.float32)
    comm.Bcast(sh_com, root = 0)
    print_status(rank, start_time, "Gotten HDF5 raw data. Number of SHs is {0}".format(sh_com.shape[0]))
    
    # Discard Shs with bad resolution
    print_status(rank, start_time, "Discarding SHs with bad resolution..")
    if rank == 0:
        sh_idx = 0
        while sh_idx < len(sh_cat):
            if len(sh_cat[sh_idx]) < config.MIN_NUMBER_DM_PTCS:
                sh_cat =  [x if idx != sh_idx else [] for idx, x in enumerate(sh_cat)]  # Remove Sh that has bad resolution
            sh_idx += 1
    sh_cat = comm.bcast(sh_cat, root = 0)
    print_status(rank, start_time, "The number of valid shs after discarding low-resolution ones is {0}".format(np.array([0 for x in sh_cat if x != []]).shape[0]))
    
    # Defining galaxies: Method 1: 1 sh = at most 1 galaxy
    print_status(rank, start_time, "Creating Gx CAT..")
    nb_jobs_to_do = star_xyz.shape[0]
    perrank = nb_jobs_to_do//size + (nb_jobs_to_do//size == 0)*1
    do_sth = rank <= nb_jobs_to_do-1
    comm.Barrier()
    gx_cat = [[] for i in range(sh_com.shape[0])]
    if size <= nb_jobs_to_do:
        last = rank == size - 1 # Whether or not last process
    else:
        last = rank == nb_jobs_to_do - 1
    discard_sh = 0
    for star_ptc in range(rank*perrank, rank*perrank+do_sth*(perrank+last*(nb_jobs_to_do-(rank+1)*perrank))):
        dist_x = abs(star_xyz[star_ptc,0]-sh_com[:,0])
        dist_x[dist_x > config.L_BOX/2] = config.L_BOX-dist_x[dist_x > config.L_BOX/2]
        dist_y = abs(star_xyz[star_ptc,1]-sh_com[:,1])
        dist_y[dist_y > config.L_BOX/2] = config.L_BOX-dist_y[dist_y > config.L_BOX/2]
        dist_z = abs(star_xyz[star_ptc,2]-sh_com[:,2])
        dist_z[dist_z > config.L_BOX/2] = config.L_BOX-dist_z[dist_z > config.L_BOX/2]
        argmin = np.argmin(dist_x**2+dist_y**2+dist_z**2)
        # 1st round (no discarding): If argmin is a poorly resolved sh (= most likely very low mass) don't discard star_ptc, just keep track
        if sh_cat[argmin] == []:
            discard_sh += 1 # In misalignment study, such a gx will just yield nan misalignment angle.
        gx_cat[argmin].append(star_ptc)
    gx_cat = comm.gather(gx_cat, root = 0)
    discard_sh = comm.reduce(discard_sh, op=MPI.SUM, root = 0)
    if rank == 0:
        gx_cat_full = [[] for i in range(sh_com.shape[0])]
        for r in range(size):
            for h_idx in range(sh_com.shape[0]):
                gx_cat_full[h_idx] += gx_cat[r][h_idx]
    else:
        gx_cat_full = None
    gx_cat = comm.bcast(gx_cat_full, root = 0)
    print_status(rank, start_time, "Gotten Gx CAT. The number of valid gxs is {0}. Out of {1}, we noted {2} star particles whose parent SH is poorly resolved".format(np.array([0 for x in gx_cat if x != []]).shape[0], star_xyz.shape[0], discard_sh))
    
    # Storing gx catalogue
    with open('{0}/gx_cat_fdm_{1}.txt'.format(config.CAT_DEST, config.SNAP), 'w') as filehandle:
        json.dump(gx_cat, filehandle)
    
    # Discard gxs with bad resolution
    if rank == 0:
        gx_idx = 0
        while gx_idx < len(gx_cat):
            if len(gx_cat[gx_idx]) < config.MIN_NUMBER_STAR_PTCS:
                gx_cat = [x for idx, x in enumerate(gx_cat) if idx != gx_idx] # Remove gx that has bad resolution
            else:
                gx_idx += 1
    gx_cat = comm.bcast(gx_cat, root = 0)
    print_status(rank, start_time, "The number of valid gxs after discarding low-resolution ones is {0}".format(np.array([0 for x in gx_cat if x != []]).shape[0]))
    
    # Retrieve 9*eps with gx_cat
    if rank == 0:
        eps_9 = h5py.File(r'{0}/snap_{1}.{2}.hdf5'.format(config.HDF5_SNAP_DEST, config.SNAP, 0), 'r')['Parameters'].attrs['SofteningComovingType1']/1000*9 # In cMpc/h
        np.savetxt('{0}/eps_9_fdm_gx_{1}.txt'.format(config.CAT_DEST, config.SNAP), np.array([eps_9]), fmt='%1.7e')
    
    # Local Morphology
    print_status(rank, start_time, "Calculating local morphologies. The average number of ptcs in the gxs is {0}".format(np.average(np.array(list(map(lambda x: len([x for x in gx_cat if x != []][x]), range(len([x for x in gx_cat if x != []]))))))))
    d, q, s, minor, inter, major, gx_com, gx_m, succeeded = getMorphologies(star_xyz, gx_cat, star_masses, rdelta, "gxs", "local", start_time)
    print_status(rank, start_time, "Gotten local morphologies")
    
    if rank == 0:
        if d != []:
            d = np.reshape(np.array(d), (np.array(d).shape[0], np.array(d).shape[1])) # Has shape (number_of_gxs, d_discr), each d_discr a float
            q = np.reshape(np.array(q), (np.array(q).shape[0], np.array(q).shape[1])) # Has shape (number_of_gxs, d_discr), each d_discr a float
            s = np.reshape(np.array(s), (np.array(s).shape[0], np.array(s).shape[1])) # Has shape (number_of_gxs, d_discr), each d_discr a float
            minor = np.array(minor) # Has shape (number_of_gxs, d_discr), each of (3,) shape
            inter = np.array(inter) # Has shape (number_of_gxs, d_discr), each of (3,) shape
            major = np.array(major) # Has shape (number_of_gxs, d_discr), each of (3,) shape
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
        
        else:
            d = np.array(d) # Empty array
            q = np.array(q) # Empty
            s = np.array(s) # Empty
            minor_new_re = np.array(minor) 
            inter_new_re = np.array(inter) 
            major_new_re = np.array(major) 
        gx_com = np.array(gx_com) # Has shape (number_of_gxs, 3)
        gx_m = np.array(gx_m) # Has shape (number_of_gxs, )
        
        # Create catalogue storing major axes at maximal r_ell and gx_com
        a_com_cat_gx = [[] for i in range(d.shape[0])] # For each gx, 6 floats, first 3 give orientation of major axis at maximal r_ell, last 3 give gx_com's x, y, z
        
        for gx in range(d.shape[0]):
            a_com_cat_gx[gx].extend((major[gx][-1][0], major[gx][-1][1], major[gx][-1][2], gx_com[gx][0], gx_com[gx][1], gx_com[gx][2], gx_m[gx]))
            
        cat_local = deepcopy(gx_cat)
        for gx_idx in range(len(cat_local)):
            if gx_idx not in succeeded:
                cat_local =  [x if idx != gx_idx else [] for idx, x in enumerate(cat_local)]
        with open('{0}/gx_cat_local_fdm_{1}.txt'.format(config.CAT_DEST, config.SNAP), 'w') as filehandle:
            json.dump(cat_local, filehandle)
        
        # Writing
        with open('{0}/a_com_cat_local_fdm_gx_{1}.txt'.format(config.CAT_DEST, config.SNAP), 'w') as filehandle:
            json.dump(a_com_cat_gx, filehandle)
       
        # Storing np.arrays
        np.savetxt('{0}/d_local_fdm_gx_{1}.txt'.format(config.CAT_DEST, config.SNAP), d, fmt='%1.7e')
        np.savetxt('{0}/q_local_fdm_gx_{1}.txt'.format(config.CAT_DEST, config.SNAP), q, fmt='%1.7e')
        np.savetxt('{0}/s_local_fdm_gx_{1}.txt'.format(config.CAT_DEST, config.SNAP), s, fmt='%1.7e')
        np.savetxt('{0}/minor_local_fdm_gx_{1}.txt'.format(config.CAT_DEST, config.SNAP), minor_new_re, fmt='%1.7e')
        np.savetxt('{0}/inter_local_fdm_gx_{1}.txt'.format(config.CAT_DEST, config.SNAP), inter_new_re, fmt='%1.7e')
        np.savetxt('{0}/major_local_fdm_gx_{1}.txt'.format(config.CAT_DEST, config.SNAP), major_new_re, fmt='%1.7e')
        
    # Morphology: Overall Shape (with E1 at large radius)
    print_status(rank, start_time, "Calculating overall morphologies. The average number of ptcs in the gxs is {0}".format(np.average(np.array(list(map(lambda x: len([x for x in gx_cat if x != []][x]), range(len([x for x in gx_cat if x != []]))))))))
    d, q, s, minor, inter, major, gxs_com, gx_m = getMorphologies(star_xyz, gx_cat, star_masses, rdelta, "gxs", "overall", start_time)
    print_status(rank, start_time, "Gotten morphologies")
    
    if rank == 0:
        if d != []:
            d = np.reshape(np.array(d), (np.array(d).shape[0], np.array(d).shape[1])) # Has shape (number_of_gxs, 1), each d_discr a float
            q = np.reshape(np.array(q), (np.array(q).shape[0], np.array(q).shape[1])) # Has shape (number_of_gxs, 1), each d_discr a float
            s = np.reshape(np.array(s), (np.array(s).shape[0], np.array(s).shape[1])) # Has shape (number_of_gxs, 1), each d_discr a float
            minor = np.array(minor) # Has shape (number_of_gxs, 1), each of (3,) shape
            inter = np.array(inter) # Has shape (number_of_gxs, 1), each of (3,) shape
            major = np.array(major) # Has shape (number_of_gxs, 1), each of (3,) shape
            major_new = np.zeros((major.shape[0], major.shape[1], 3)) # Has shape (number_of_gxs, 1, 3) finally
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
        
        else:
            d = np.array(d) # Empty array
            q = np.array(q) # Empty
            s = np.array(s) # Empty
            minor_new_re = np.array(minor) 
            inter_new_re = np.array(inter) 
            major_new_re = np.array(major) 
        gxs_com = np.array(gxs_com) # Has shape (number_of_gxs, 3)
        gx_m = np.array(gx_m) # Has shape (number_of_gxs, )
        
        # Create catalogue storing major axes at maximal r_ell and gx_com
        a_com_cat = [[] for i in range(d.shape[0])] # For each gx, 6 floats, first 3 give orientation of major axis at maximal r_ell, last 3 give gx_com's x, y, z
        
        for gx in range(d.shape[0]):
            a_com_cat[gx].extend((major[gx][-1][0], major[gx][-1][1], major[gx][-1][2], gxs_com[gx][0], gxs_com[gx][1], gxs_com[gx][2], gx_m[gx]))
                
        # Writing
        with open('{0}/a_com_cat_overall_fdm_gx_{1}.txt'.format(config.CAT_DEST, config.SNAP), 'w') as filehandle:
            json.dump(a_com_cat, filehandle)
        
        with open('{0}/gx_cat_overall_fdm_{1}.txt'.format(config.CAT_DEST, config.SNAP), 'w') as filehandle:
            json.dump(gx_cat, filehandle)
            
        # Storing np.arrays
        np.savetxt('{0}/d_overall_fdm_gx_{1}.txt'.format(config.CAT_DEST, config.SNAP), d, fmt='%1.7e')
        np.savetxt('{0}/q_overall_fdm_gx_{1}.txt'.format(config.CAT_DEST, config.SNAP), q, fmt='%1.7e')
        np.savetxt('{0}/s_overall_fdm_gx_{1}.txt'.format(config.CAT_DEST, config.SNAP), s, fmt='%1.7e')
        np.savetxt('{0}/minor_overall_fdm_gx_{1}.txt'.format(config.CAT_DEST, config.SNAP), minor_new_re, fmt='%1.7e')
        np.savetxt('{0}/inter_overall_fdm_gx_{1}.txt'.format(config.CAT_DEST, config.SNAP), inter_new_re, fmt='%1.7e')
        np.savetxt('{0}/major_overall_fdm_gx_{1}.txt'.format(config.CAT_DEST, config.SNAP), major_new_re, fmt='%1.7e')