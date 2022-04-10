#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 11:10:42 2021

@author: tibor
"""

import json
import numpy as np
import os
import argparse
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.join(currentdir, '..', '..', '..', 'config'))
import config
from config import makeGlobalDM_TYPE
config.initialize()
sys.path.append(os.path.join(currentdir, '..', 'utilities'))
from print_msg import print_status
from mpi4py import MPI
import time
start_time = time.time()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

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


def extractSpine():
    
    # Housekeeping
    print_status(rank,start_time,'Starting extractSpine()')
    
    # Import skeleton data in ASCII format 
    if rank == 0:
        with open('DM_{0}_{1}.NDnet.up.NDskl.a.NDskl'.format(dm_type, snap), 'r') as file:
            data = file.readlines()
       
        print_status(rank, start_time, 'Finding starting and ending line')
        for n, line in enumerate(data):
            if line.split()[0] == '[FILAMENTS]': # .split splits at every empty space in this case
                nstart = n
            if "nstart" in locals():
                if line.split()[0] == '[CRITICAL':
                    nend = n
        data = data[nstart+1:nend] # Go directly to first "nfil" line
    else:
        data = None
    data = comm.bcast(data, root = 0)
    print_status(rank,start_time,'Finished combining into array of strings.')
    
    # Find number of filaments
    nb_fils = np.asarray(list(map(int, data[0].split())))[0]
    print_status(rank,start_time,'The number of filaments is {0}.'.format(nb_fils))
    
    # Advance by one to go directly to first "CP1 CP2 nSamp"
    data = data[1:]
    nb_jobs_to_do = len(data)
        
    # Find sampling positions and add to sampling_pos list
    perrank = nb_jobs_to_do//size + (nb_jobs_to_do//size == 0)*1 # Again: nb_jobs_to_do//size == 0 in case size > nb_jobs_to_do
    do_sth = rank <= nb_jobs_to_do-1
    if size <= nb_jobs_to_do:
        last = rank == size - 1 # Whether or not last process
    else:
        last = rank == nb_jobs_to_do - 1
    first = rank == 0 # Whether or not first process
    comm.Barrier()
    count = 0
    sampling_pos = []
    if do_sth:
        if first:
            correct_left = 0
        else:
            found = False
            run = 0
            while not found:
                if data[perrank*rank+run][0] != " ": # We have found new filament section: "CP1 CP2 nSamp" string is preceded by a blank space character
                    found = True
                else:
                    run += 1
            correct_left = run
        if last:
            correct_right = 0
        else:
            found = False
            run = 0
            while not found:
                if data[perrank*(rank+1)+run][0] != " ":
                    found = True
                else:
                    run += 1
            correct_right = run
        fil_section_rank = data[perrank*rank+correct_left:perrank*(rank+1)+last*(len(data)-(rank+1)*perrank)+correct_right]
        n = 0
        line = fil_section_rank[n]
        while line:
            sp_num = np.asarray(list(map(int, line.split())))[2] # Number of sampling points for filament
            samples = []
            n += 1; count += 1
            last_one = sp_num == len(fil_section_rank[n:]) # Last filament in fil_section_rank or not
            for line in fil_section_rank[n:n+sp_num]:
                toadd = np.asarray(list(map(float, line.split()))) # Extract sampling point coordinates
                samples.append(list(toadd*config.FAST_DISPERSE)) # Careful: If mse was fed with output from delaunay cmd, then remove config.FAST_DISPERSE.
            sampling_pos.append(samples)
            if last_one:
                break
            else:
                n += sp_num
                line = fil_section_rank[n]
    comm.Barrier()
    sampling_pos = comm.gather(sampling_pos, root = 0) # List of lists of lists of lists
    count_new = comm.gather(count, root=0)
    if rank == 0:
        sampling_pos = [sampling_pos[i][j] for i in range(size) for j in range(count_new[i])]
    sampling_pos = comm.bcast(sampling_pos, root = 0) # List of lists of lists
    print_status(rank,start_time,'Finished sampling position parsing.')
    
    if rank == 0:
        with open('sampling_pos_{0}_{1}.txt'.format(dm_type, snap), 'w') as filehandle:
            json.dump(sampling_pos, filehandle)
    print_status(rank,start_time,'Dumped sampling point positions into sampling_pos_{0}_{1}.txt.'.format(dm_type, snap))
            
    print_status(rank,start_time,'Start constructing 4D samples array')
    nb_jobs_to_do = len(sampling_pos)
    perrank = nb_jobs_to_do//size + (nb_jobs_to_do//size == 0)*1
    do_sth = rank <= nb_jobs_to_do-1
    if size <= nb_jobs_to_do:
        last = rank == size - 1 # Whether or not last process
    else:
        last = rank == nb_jobs_to_do - 1
    comm.Barrier()
    count = 0
    correct = perrank*rank
    samples_x = np.empty(0, dtype = np.float32)
    samples_y = np.empty(0, dtype = np.float32)
    samples_z = np.empty(0, dtype = np.float32)
    samples_fil = np.empty(0, dtype = np.float32)
    sampling_pos_rank = sampling_pos[rank*perrank:rank*perrank+do_sth*(perrank+last*(nb_jobs_to_do-(rank+1)*perrank))]
    for fil in range(len(sampling_pos_rank)):
        for j in range(len(sampling_pos_rank[fil])):
            samples_x = np.hstack((samples_x, np.float32(sampling_pos_rank[fil][j][0])))
            samples_y = np.hstack((samples_y, np.float32(sampling_pos_rank[fil][j][1])))
            samples_z = np.hstack((samples_z, np.float32(sampling_pos_rank[fil][j][2])))
            samples_fil = np.hstack((samples_fil, np.float32(fil+correct)))
            count += 1
    
    count_new = comm.gather(count, root=0)
    count_new = comm.bcast(count_new, root = 0)
    nb_pts = np.sum(np.array(count_new))
    recvcounts = np.array(count_new)
    rdispls = np.zeros_like(recvcounts)
    for j in range(rdispls.shape[0]):
        rdispls[j] = np.sum(recvcounts[:j])
    samples_x_total = np.empty(nb_pts, dtype = np.float32)
    samples_y_total = np.empty(nb_pts, dtype = np.float32)
    samples_z_total = np.empty(nb_pts, dtype = np.float32)
    samples_fil_total = np.empty(nb_pts, dtype = np.float32)
    comm.Barrier()
    
    comm.Gatherv(samples_x, [samples_x_total, recvcounts, rdispls, MPI.FLOAT], root = 0)
    comm.Gatherv(samples_y, [samples_y_total, recvcounts, rdispls, MPI.FLOAT], root = 0)
    comm.Gatherv(samples_z, [samples_z_total, recvcounts, rdispls, MPI.FLOAT], root = 0)
    comm.Gatherv(samples_fil, [samples_fil_total, recvcounts, rdispls, MPI.FLOAT], root = 0)
    
    samples = np.hstack((np.reshape(samples_x_total, (samples_x_total.shape[0],1)), np.reshape(samples_y_total, (samples_y_total.shape[0],1)), np.reshape(samples_z_total, (samples_z_total.shape[0],1)), np.reshape(samples_fil_total, (samples_fil_total.shape[0],1)))) # 4D array, only available in rank = 0
    print_status(rank,start_time,'Constructed 4D samples array')
    if rank == 0:
        np.savetxt('samples_{0}_{1}.txt'.format(dm_type, snap), samples, fmt='%1.7e')
    print_status(rank,start_time,'Dumped 4D samples array')
    
extractSpine()