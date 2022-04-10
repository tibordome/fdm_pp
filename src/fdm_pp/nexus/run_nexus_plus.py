#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 18:23:43 2021

@author: tibor
"""

import time
start_time = time.time()
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.join(currentdir, '..', '..', '..', 'config'))
sys.path.append(os.path.join(currentdir, '..', 'utilities'))
from print_msg import print_status
import numpy as np
import config
from config import makeGlobalDM_TYPE, makeGlobalRSMOOTH, getA
config.initialize()
from calc_sigs_nexus_plus import getSigsNexusPlus
rank = 0

print_status(rank, start_time, "Running nexus_plus signature calculation script with N = {0}, L = {1}, SNAP_ABB = {2}".format(config.N, config.L_BOX, config.SNAP_ABB))

# Get Nexus catalogues
for snap in config.SNAP_ABB:
    
    makeGlobalDM_TYPE('fdm', snap, start_time)
    Z = getA()**(-1)-1
    makeGlobalRSMOOTH(np.array([np.sqrt(2)**n*0.5/(1+Z) for n in range(8)]).astype('float32'))
    getSigsNexusPlus(start_time)