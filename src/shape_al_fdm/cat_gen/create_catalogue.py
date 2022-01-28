#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 18:23:43 2021

@author: tibor
"""

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
import time
start_time = time.time()
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.join(currentdir, '..', '..', '..', 'config'))
import config
config.initialize()
from create_cat_dm import createCatDM
from create_cat_major_com_dm import createCatMajorCOMDM
from create_cat_major_com_gx import createCatMajorCOMGx


# DM Catalogue generation
createCatDM()
createCatMajorCOMDM()

# Gxs Catalogue generation
createCatDM()
createCatMajorCOMGx()