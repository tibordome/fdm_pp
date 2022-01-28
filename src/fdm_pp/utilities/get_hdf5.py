#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 13:17:44 2021

@author: tibor
"""

import numpy as np
import h5py
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
import config

def getHDF5DMData():
        
    dm_masses = np.empty(0, dtype = np.float32)
    perrank = config.SNAP_MAX//size
    count = 0
    last = rank == size - 1 # Whether or not last process
    
    xlin = np.arange(0, config.N)*config.L_BOX/config.N
    dm_x,dm_y,dm_z = np.meshgrid(xlin, xlin, xlin, indexing='ij')
    
    for snap_run in range(rank*perrank, (rank+1)*perrank+last*(config.SNAP_MAX-(rank+1)*perrank)):
        f = h5py.File(r'{0}/snap_{1}.{2}.hdf5'.format(config.HDF5_SNAP_DEST, config.SNAP_ABB, snap_run), 'r')
        dm_masses = np.hstack((dm_masses, f['PartType1/Masses'][:]))
        count += f['PartType1/Coordinates'][:].shape[0]
    count_new = comm.gather(count, root=0)
    count_new = comm.bcast(count_new, root = 0)
    nb_dm_ptcs = np.sum(np.array(count_new))
    comm.Barrier()
    recvcounts = np.array(count_new)
    rdispls = np.zeros_like(recvcounts)
    for j in range(rdispls.shape[0]):
        rdispls[j] = np.sum(recvcounts[:j])
    dm_masses_total = np.empty(nb_dm_ptcs, dtype = np.float32)
    comm.Gatherv(dm_masses, [dm_masses_total, recvcounts, rdispls, MPI.FLOAT], root = 0)

    pieces = 1 + (nb_dm_ptcs>=3*10**8)*nb_dm_ptcs//(3*10**8) # Not too high since this is a slow-down!
    chunk = nb_dm_ptcs//pieces
    dm_masses = np.empty(0, dtype = np.float32)
    for i in range(pieces):
        to_bcast = dm_masses_total[i*chunk:(i+1)*chunk+(i==(pieces-1))*(nb_dm_ptcs-pieces*chunk)]
        comm.Bcast(to_bcast, root=0)
        dm_masses = np.hstack((dm_masses, to_bcast))
    
    dm_xyz = np.hstack((np.reshape(dm_x, (dm_x.shape[0],1)), np.reshape(dm_y, (dm_y.shape[0],1)), np.reshape(dm_z, (dm_z.shape[0],1))))

    return dm_xyz, dm_masses

def getHDF5DMStarData():
    
    star_x = np.empty(0, dtype = np.float32)
    star_y = np.empty(0, dtype = np.float32)
    star_z = np.empty(0, dtype = np.float32)
    dm_masses = np.empty(0, dtype = np.float32)
    star_masses = np.empty(0, dtype = np.float32)
    perrank = config.SNAP_MAX//size
    count_dm = 0
    count_star = 0
    last = rank == size - 1 # Whether or not last process
    for snap_run in range(rank*perrank, (rank+1)*perrank+last*(config.SNAP_MAX-(rank+1)*perrank)):
        f = h5py.File(r'{0}/snap_{1}.{2}.hdf5'.format(config.HDF5_SNAP_DEST, config.SNAP_ABB, snap_run), 'r')
        dm_masses = np.hstack((dm_masses, f['PartType1/Masses'][:]))
        count_dm += f['PartType1/Coordinates'][:].shape[0]
        if 'PartType4/Coordinates' in f:
            star_x = np.hstack((star_x, np.float32(f['PartType4/Coordinates'][:,0]/1000))) # in cMpc/h = 3.085678e+27 cm
            star_y = np.hstack((star_y, np.float32(f['PartType4/Coordinates'][:,1]/1000)))
            star_z = np.hstack((star_z, np.float32(f['PartType4/Coordinates'][:,2]/1000)))
            star_masses = np.hstack((star_masses, f['PartType4/Masses'][:]))
            count_star += f['PartType4/Coordinates'][:].shape[0]
    count_new_dm = comm.gather(count_dm, root=0)
    count_new_dm = comm.bcast(count_new_dm, root = 0)
    nb_dm_ptcs = np.sum(np.array(count_new_dm))
    count_new_star = comm.gather(count_star, root=0)
    count_new_star = comm.bcast(count_new_star, root = 0)
    nb_star_ptcs = np.sum(np.array(count_new_star))
    comm.Barrier()
    
    recvcounts_dm = np.array(count_new_dm)
    rdispls_dm = np.zeros_like(recvcounts_dm)
    for j in range(rdispls_dm.shape[0]):
        rdispls_dm[j] = np.sum(recvcounts_dm[:j])
    recvcounts_star = np.array(count_new_star)
    rdispls_star = np.zeros_like(recvcounts_star)
    for j in range(rdispls_star.shape[0]):
        rdispls_star[j] = np.sum(recvcounts_star[:j])
    dm_masses_total = np.empty(nb_dm_ptcs, dtype = np.float32)
    star_x_total = np.empty(nb_star_ptcs, dtype = np.float32)
    star_y_total = np.empty(nb_star_ptcs, dtype = np.float32)
    star_z_total = np.empty(nb_star_ptcs, dtype = np.float32)
    star_masses_total = np.empty(nb_star_ptcs, dtype = np.float32)
    
    comm.Gatherv(dm_masses, [dm_masses_total, recvcounts_dm, rdispls_dm, MPI.FLOAT], root = 0)

    comm.Gatherv(star_x, [star_x_total, recvcounts_star, rdispls_star, MPI.FLOAT], root = 0)
    comm.Gatherv(star_y, [star_y_total, recvcounts_star, rdispls_star, MPI.FLOAT], root = 0)
    comm.Gatherv(star_z, [star_z_total, recvcounts_star, rdispls_star, MPI.FLOAT], root = 0)
    comm.Gatherv(star_masses, [star_masses_total, recvcounts_star, rdispls_star, MPI.FLOAT], root = 0)

    pieces = 1 + (nb_dm_ptcs>=3*10**8)*nb_dm_ptcs//(3*10**8) # Not too high since this is a slow-down!
    chunk = nb_dm_ptcs//pieces
    dm_masses = np.empty(0, dtype = np.float32)
    for i in range(pieces):
        to_bcast = dm_masses_total[i*chunk:(i+1)*chunk+(i==(pieces-1))*(nb_dm_ptcs-pieces*chunk)]
        comm.Bcast(to_bcast, root=0)
        dm_masses = np.hstack((dm_masses, to_bcast))
    
    xlin = np.arange(0, config.N)*config.L_BOX/config.N
    dm_x,dm_y,dm_z = np.meshgrid(xlin, xlin, xlin, indexing='ij')
    dm_xyz = np.hstack((np.reshape(dm_x, (dm_x.shape[0],1)), np.reshape(dm_y, (dm_y.shape[0],1)), np.reshape(dm_z, (dm_z.shape[0],1))))
    
    pieces = 1 + (nb_star_ptcs>=3*10**8)*nb_star_ptcs//(3*10**8) # Not too high since this is a slow-down!
    chunk = nb_star_ptcs//pieces
    star_x = np.empty(0, dtype = np.float32)
    star_y = np.empty(0, dtype = np.float32)
    star_z = np.empty(0, dtype = np.float32)
    star_masses = np.empty(0, dtype = np.float32)
    for i in range(pieces):
        to_bcast = star_x_total[i*chunk:(i+1)*chunk+(i==(pieces-1))*(nb_star_ptcs-pieces*chunk)]
        comm.Bcast(to_bcast, root=0)
        star_x = np.hstack((star_x, to_bcast))
        to_bcast = star_y_total[i*chunk:(i+1)*chunk+(i==(pieces-1))*(nb_star_ptcs-pieces*chunk)]
        comm.Bcast(to_bcast, root=0)
        star_y = np.hstack((star_y, to_bcast))
        to_bcast = star_z_total[i*chunk:(i+1)*chunk+(i==(pieces-1))*(nb_star_ptcs-pieces*chunk)]
        comm.Bcast(to_bcast, root=0)
        star_z = np.hstack((star_z, to_bcast))
        to_bcast = star_masses_total[i*chunk:(i+1)*chunk+(i==(pieces-1))*(nb_star_ptcs-pieces*chunk)]
        comm.Bcast(to_bcast, root=0)
        star_masses = np.hstack((star_masses, to_bcast))
    
    star_xyz = np.hstack((np.reshape(star_x, (star_x.shape[0],1)), np.reshape(star_y, (star_y.shape[0],1)), np.reshape(star_z, (star_z.shape[0],1))))

    return dm_xyz, dm_masses, star_xyz, star_masses

def getHDF5StarData():
     
    star_x = np.empty(0, dtype = np.float32)
    star_y = np.empty(0, dtype = np.float32)
    star_z = np.empty(0, dtype = np.float32)
    star_masses = np.empty(0, dtype = np.float32)
    perrank = config.SNAP_MAX//size
    count_star = 0
    last = rank == size - 1 # Whether or not last process
    for snap_run in range(rank*perrank, (rank+1)*perrank+last*(config.SNAP_MAX-(rank+1)*perrank)):
        f = h5py.File(r'{0}/snap_{1}.{2}.hdf5'.format(config.HDF5_SNAP_DEST, config.SNAP_ABB, snap_run), 'r')
        if 'PartType4/Coordinates' in f:
            star_x = np.hstack((star_x, np.float32(f['PartType4/Coordinates'][:,0]/1000))) # in cMpc/h = 3.085678e+27 cm
            star_y = np.hstack((star_y, np.float32(f['PartType4/Coordinates'][:,1]/1000)))
            star_z = np.hstack((star_z, np.float32(f['PartType4/Coordinates'][:,2]/1000)))
            star_masses = np.hstack((star_masses, f['PartType4/Masses'][:]))
            count_star += f['PartType4/Coordinates'][:].shape[0]
    count_new_star = comm.gather(count_star, root=0)
    count_new_star = comm.bcast(count_new_star, root = 0)
    nb_star_ptcs = np.sum(np.array(count_new_star))
    comm.Barrier()
    
    recvcounts_star = np.array(count_new_star)
    rdispls_star = np.zeros_like(recvcounts_star)
    for j in range(rdispls_star.shape[0]):
        rdispls_star[j] = np.sum(recvcounts_star[:j])
    star_x_total = np.empty(nb_star_ptcs, dtype = np.float32)
    star_y_total = np.empty(nb_star_ptcs, dtype = np.float32)
    star_z_total = np.empty(nb_star_ptcs, dtype = np.float32)
    star_masses_total = np.empty(nb_star_ptcs, dtype = np.float32)
    
    comm.Gatherv(star_x, [star_x_total, recvcounts_star, rdispls_star, MPI.FLOAT], root = 0)
    comm.Gatherv(star_y, [star_y_total, recvcounts_star, rdispls_star, MPI.FLOAT], root = 0)
    comm.Gatherv(star_z, [star_z_total, recvcounts_star, rdispls_star, MPI.FLOAT], root = 0)
    comm.Gatherv(star_masses, [star_masses_total, recvcounts_star, rdispls_star, MPI.FLOAT], root = 0)

    pieces = 1 + (nb_star_ptcs>=3*10**8)*nb_star_ptcs//(3*10**8) # Not too high since this is a slow-down!
    chunk = nb_star_ptcs//pieces
    star_x = np.empty(0, dtype = np.float32)
    star_y = np.empty(0, dtype = np.float32)
    star_z = np.empty(0, dtype = np.float32)
    star_masses = np.empty(0, dtype = np.float32)
    for i in range(pieces):
        to_bcast = star_x_total[i*chunk:(i+1)*chunk+(i==(pieces-1))*(nb_star_ptcs-pieces*chunk)]
        comm.Bcast(to_bcast, root=0)
        star_x = np.hstack((star_x, to_bcast))
        to_bcast = star_y_total[i*chunk:(i+1)*chunk+(i==(pieces-1))*(nb_star_ptcs-pieces*chunk)]
        comm.Bcast(to_bcast, root=0)
        star_y = np.hstack((star_y, to_bcast))
        to_bcast = star_z_total[i*chunk:(i+1)*chunk+(i==(pieces-1))*(nb_star_ptcs-pieces*chunk)]
        comm.Bcast(to_bcast, root=0)
        star_z = np.hstack((star_z, to_bcast))
        to_bcast = star_masses_total[i*chunk:(i+1)*chunk+(i==(pieces-1))*(nb_star_ptcs-pieces*chunk)]
        comm.Bcast(to_bcast, root=0)
        star_masses = np.hstack((star_masses, to_bcast))
    
    star_xyz = np.hstack((np.reshape(star_x, (star_x.shape[0],1)), np.reshape(star_y, (star_y.shape[0],1)), np.reshape(star_z, (star_z.shape[0],1))))
   
    return star_xyz, star_masses