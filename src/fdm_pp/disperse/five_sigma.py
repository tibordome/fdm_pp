#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 12:44:02 2021

@author: tibor
"""

import numpy as np
from mpi4py import MPI
import time
start_time = time.time()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
import os
import argparse
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.join(currentdir, '..', 'utilities'))
from print_msg import print_status
sys.path.append(os.path.join(currentdir, '..', '..', '..', 'config'))
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

def extract_5sigma():
    
    # Housekeeping
    print_status(rank,start_time,'Starting extract_5sigma()')
    
    values = []
    five_sig = 0
    # Import critical points data
    with open('DM_{0}_{1}.NDnet.up.NDskl.a.crits'.format(dm_type, snap), 'r') as file:
        data = file.readlines()
    data = data[3:]
    for n, line in enumerate(data):
        x = np.asarray(list(map(float, line.split())))
        if int(x[4]) == 3: # Focus on nodes (crit. ind. is 3) only: Their pairs will have critical index = 2.
            pot_pair = np.asarray(list(map(float, data[int(x[5])].split())))
            if int(x[5]) != int(pot_pair[5]): # Ignore points without pairs
                assert int(pot_pair[4]) == 2 # Double-check that crit_ind 3 matches to crit_ind 2.
                values.append([x[3], pot_pair[3]])
            
    pairs_part1 = [x[0] for x in values]
    pairs_part2 = [x[1] for x in values]
    five_sig = 5*np.sqrt(np.square(np.subtract(pairs_part1,pairs_part2)).mean()) # 5*RMSE
    with open("five_sig_{0}_{1}.txt".format(dm_type, snap), 'w') as f:
        f.write("five_sig = "+str(five_sig))
    print_status(rank,start_time,"The number of filaments in the cut-0-run is {0}. 5Sigma is {1}. Written to file five_sig_{2}_{3}.txt".format(len(values), five_sig, dm_type, snap))

extract_5sigma()