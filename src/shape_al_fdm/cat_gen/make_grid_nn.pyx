#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 10:19:41 2020

@author: tibor
"""

import numpy as np
from cython import parallel
cimport openmp
cimport numpy as cnp
import sys
import math
from cython.parallel import prange
from libc.stdio cimport printf
from libc.math cimport round as cyround

def makeGridWithNNPBC(float[:] x, float[:] y, float[:] z, float[:] dm_masses, float L_BOX, int N):
    """
    Create structured data from unstructured data (discrete points with possibly unequal
    masses given) via NN algorithm
    Periodic boundary conditions (PBC) are assumed.
    Assumption: np.max(x)-np.min(x)=np.max(y)-np.min(y)=np.max(z)-np.min(z)
    Assumption: number of particles is a cube of some number, though can be relaxed
    by commenting 2nd and 3rd line in code
    Parameters
    ----------
    x, y, z : 1D arrays, (x,y,z)-coordinates of the point distribution
    dm_masses: 1D array (Float or Double), masses of DM particles
    FDM field real and imaginary parts (assigned at each FDM particle)
    Returns
    ----------
    grid : An (N, N, N)-shaped array
    A regular grid in 3D
    """
    
    # Check for consistency
    if np.min(x) < 0 or np.min(y) < 0 or np.min(z) < 0:
        sys.exit("Error! Some coordinates are negative...")
    if np.max(x) > L_BOX or np.max(y) > L_BOX or np.max(z) > L_BOX:
        sys.exit("Error! Some coordinates exceed the maximum admitted value...")
    
    # NN binnning
    cdef int ix
    cdef int iy
    cdef int iz
    cdef float[:,:,:] rhoNN = np.zeros((N,N,N), dtype=np.float32)
    cdef int len_x = len(x)
    cdef Py_ssize_t p
    for p in prange(len_x, nogil = True): # For each DM/gas particle. OpenMP decides how many threads to spawn.
        #printf("The thread ID is %d\n", parallel.threadid()) 
        if p == 0:
            printf("The number of threads employed by OpenMP in the NN grid construction is %i\n", openmp.omp_get_num_threads())
        
        # Find nearest cell
        ix = int(cyround(x[p] / L_BOX * N - 0.5)) 
        iy = int(cyround(y[p] / L_BOX * N - 0.5)) 
        iz = int(cyround(z[p] / L_BOX * N - 0.5)) 
        if ix == N:    # To avoid IndexError: We choose to put points at the positive end of x-dir into the first box (as PBC stipulates).
            ix = 0
        if iy == N:
            iy = 0
        if iz == N:
            iz = 0
        if ix == -1:   # To avoid IndexError: This does not bother us since round(-0.5)=0.0, just in case box is not [0, max]^3 but [min, max]^3
            ix = ix + 1
        if iy == -1:
            iy = iy + 1
        if iz == -1:
            iz = iz + 1
        
        rhoNN[ix,iy,iz] += dm_masses[p]/(L_BOX/N)**3
    return rhoNN.base
        
