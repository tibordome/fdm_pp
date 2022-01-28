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
from scipy.interpolate import griddata
cimport openmp
from copy import deepcopy
cimport numpy as cnp
import sys
import math
from cython.parallel import prange
from libc.stdio cimport printf
from libc.math cimport round as cyround

def makeGridWithCICPBC(float[:] x, float[:] y, float[:] z, float[:] masses, float L_BOX, int N):
    """
    Create structured data from unstructured data (discrete points with possibly unequal
    masses given) via Cloud-In-Cell
    Periodic boundary conditions (PBC) are assumed.
    Assumption: np.max(x)-np.min(x)=np.max(y)-np.min(y)=np.max(z)-np.min(z)
    Assumption: number of particles is a cube of some number, though can be relaxed
    by commenting 2nd and 3rd line in code
    Parameters
    ----------
    x, y, z : 1D arrays, (x,y,z)-coordinates of the point distribution
    masses: 1D array (Float or Double)
    Mass of the particles
    Returns
    ----------
    grid : An (len(x)**(1/3), len(x)**(1/3), len(x)**(1/3))-shaped array
    A regular grid in 3D
    """
    
    # Check for consistency
    if np.min(x) < 0 or np.min(y) < 0 or np.min(z) < 0:
        sys.exit("Error! Some coordinates are negative...")
    if np.max(x) > L_BOX or np.max(y) > L_BOX or np.max(z) > L_BOX:
        sys.exit("Error! Some coordinates exceed the maximum admitted value...")

    # Grid parameters
    cdef float delx = L_BOX/N
    cdef float vol = delx**3
    cdef float hx = delx / 2
    cdef float[:] xlin = ((np.arange(1,N+1))*delx).astype('float32')  # The cells' right edges
    [xx, yy, zz] = np.meshgrid(xlin, xlin, xlin, indexing='ij') 
    
    # CIC binnning
    cdef int ix
    cdef int iy
    cdef int iz
    cdef float[:,:,:] xx_mem = xx.astype(np.float32)
    cdef float[:,:,:] yy_mem = yy.astype(np.float32)
    cdef float[:,:,:] zz_mem = zz.astype(np.float32)
    cdef float dx
    cdef float dy
    cdef float dz
    cdef float nx
    cdef float ny
    cdef float nz
    cdef int facx
    cdef int facy
    cdef int facz
    cdef int ix1
    cdef int iy1
    cdef int iz1
    cdef float dlx
    cdef float dly
    cdef float dlz
    cdef float[:,:,:,:] rhoCIC = np.zeros((openmp.omp_get_max_threads(),N,N,N), dtype=np.float32)
    cdef int len_x = len(x)
    cdef Py_ssize_t p
    for p in prange(len_x, nogil = True): # For each DM/gas particle. OpenMP decides how many threads to spawn.
        #printf("The thread ID is %d\n", parallel.threadid()) 
        if p == 0:
            printf("The number of threads employed by OpenMP in the CIC grid construction is %i\n", openmp.omp_get_num_threads())
        
        # Find nearest cell
        ix = int(cyround(x[p] / L_BOX * N - 0.5)) 
        iy = int(cyround(y[p] / L_BOX * N - 0.5)) 
        iz = int(cyround(z[p] / L_BOX * N - 0.5)) 
        if ix == N:    # To avoid IndexError: We choose to put points at the positive end of x-dir into the last box, not the first box.
            ix = ix - 1
        if iy == N:
            iy = iy - 1
        if iz == N:
            iz = iz - 1
        if ix == -1:   # To avoid IndexError: This does not bother us since round(-0.5)=0.0, but in case box is not [0, max]^3...
            ix = ix + 1
        if iy == -1:
            iy = iy + 1
        if iz == -1:
            iz = iz + 1
        nx = xx_mem[ix,iy,iz]
        ny = yy_mem[ix,iy,iz]
        nz = zz_mem[ix,iy,iz]
        
        # Distance from particle to nearest cell's right edge
        dx = x[p] - nx
        dy = y[p] - ny
        dz = z[p] - nz
        
        # Note that mod N deals with PBC
        facx = 1
        facy = 1 
        facz = 1
        if dx < - hx:
            facx = -1
        if dy < - hx:
            facy = -1
        if dz < - hx:
            facz = -1
        ix1 = int(cyround((ix + facx) % N))
        iy1 = int(cyround((iy + facy) % N))
        iz1 = int(cyround((iz + facz) % N))
        if dx < - hx:
            dlx = (2*hx + dx)*(-1)
        else:
            dlx = dx
        if dy < - hx:
            dly = (2*hx + dy)*(-1)
        else:
            dly = dy
        if dz < - hx:
            dlz = (2*hx + dz)*(-1)
        else:
            dlz = dz
            
        # Place mass into the 8 neighboring cells with appropriate weights
        rhoCIC[openmp.omp_get_thread_num(), ix,iy,iz]    += masses[p] * (hx-dlx) * (hx-dly) * (hx-dlz) / vol**2;
        rhoCIC[openmp.omp_get_thread_num(), ix,iy1,iz]   += masses[p] * (hx+dlx) * (hx-dly) * (hx-dlz) / vol**2;
        rhoCIC[openmp.omp_get_thread_num(), ix1,iy,iz]   += masses[p] * (hx-dlx) * (hx+dly) * (hx-dlz) / vol**2;
        rhoCIC[openmp.omp_get_thread_num(), ix,iy,iz1]   += masses[p] * (hx-dlx) * (hx-dly) * (hx+dlz) / vol**2;
        rhoCIC[openmp.omp_get_thread_num(), ix1,iy1,iz]  += masses[p] * (hx+dlx) * (hx+dly) * (hx-dlz) / vol**2;
        rhoCIC[openmp.omp_get_thread_num(), ix1,iy,iz1]  += masses[p] * (hx-dlx) * (hx+dly) * (hx+dlz) / vol**2;
        rhoCIC[openmp.omp_get_thread_num(), ix,iy1,iz1]  += masses[p] * (hx+dlx) * (hx-dly) * (hx+dlz) / vol**2;
        rhoCIC[openmp.omp_get_thread_num(), ix1,iy1,iz1] += masses[p] * (hx+dlx) * (hx+dly) * (hx+dlz) / vol**2;
    rho = np.zeros((N, N, N), dtype = np.float32)
    for i in range(openmp.omp_get_max_threads()):
        rho += rhoCIC[i]
    return rho # Has (N1,N2,N3)-kind of ordering. No transpose needed.

def makeGridWithCICPBCUnequal(float[:] x, float[:] y, float[:] z, float[:] masses, float L_x, float L_y, float L_z, int N_x, int N_y, int N_z):
    """
    Create structured data from unstructured data (discrete points with possibly unequal
    masses given) via Cloud-In-Cell
    Periodic boundary conditions (PBC) are assumed, even though input might not fully obey this.
    Parameters
    ----------
    x, y, z : 1D arrays, (x,y,z)-coordinates of the point distribution
    masses: 1D array (Float or Double)
    Mass of the particles
    Returns
    ----------
    grid : An (N_x, N_y, N_z)-shaped array
    A regular grid in 3D
    """
    
    # Check for consistency
    if np.min(x) < 0 or np.min(y) < 0 or np.min(z) < 0:
        sys.exit("Error! Some coordinates are negative...")
    if np.max(x) > L_x or np.max(y) > L_y or np.max(z) > L_z:
        sys.exit("Error! Some coordinates exceed the maximum admitted value...")
    assert x.shape[0] == y.shape[0] and y.shape[0] == z.shape[0] and z.shape[0] == masses.shape[0], \
        'Shape of x and/or y and/or z and/or masses is wrong.'

    # Grid parameters
    cdef float delx = L_x/N_x
    cdef float dely = L_y/N_y
    cdef float delz = L_z/N_z
    cdef float volx = delx**3
    cdef float voly = dely**3
    cdef float volz = delz**3
    cdef float hx = delx / 2
    cdef float hy = dely / 2
    cdef float hz = delz / 2
    cdef float[:] xlin = ((np.arange(1,N_x+1))*delx).astype('float32')  # The cells' right edges
    cdef float[:] ylin = ((np.arange(1,N_y+1))*dely).astype('float32')  # The cells' right edges
    cdef float[:] zlin = ((np.arange(1,N_z+1))*delz).astype('float32')  # The cells' right edges
    [xx, yy, zz] = np.meshgrid(xlin, ylin, zlin, indexing='ij') 
    
    # CIC binnning
    cdef int ix
    cdef int iy
    cdef int iz
    cdef float[:,:,:] xx_mem = xx.astype(np.float32)
    cdef float[:,:,:] yy_mem = yy.astype(np.float32)
    cdef float[:,:,:] zz_mem = zz.astype(np.float32)
    cdef float dx
    cdef float dy
    cdef float dz
    cdef float nx
    cdef float ny
    cdef float nz
    cdef int facx
    cdef int facy
    cdef int facz
    cdef int ix1
    cdef int iy1
    cdef int iz1
    cdef float dlx
    cdef float dly
    cdef float dlz
    cdef float[:,:,:,:] rhoCIC = np.zeros((openmp.omp_get_max_threads(),N_x,N_y,N_z), dtype=np.float32)
    cdef int len_x = len(x)
    cdef Py_ssize_t p
    for p in prange(len_x, nogil = True): # For each DM/gas particle. OpenMP decides how many threads to spawn.
        #printf("The thread ID is %d\n", parallel.threadid()) 
        if p == 0:
            printf("The number of threads employed by OpenMP in the CIC grid construction is %i\n", openmp.omp_get_num_threads())
        
        # Find nearest cell
        ix = int(cyround(x[p] / L_x * N_x - 0.5)) 
        iy = int(cyround(y[p] / L_y * N_y - 0.5)) 
        iz = int(cyround(z[p] / L_z * N_z - 0.5)) 
        if ix == N_x:    # To avoid IndexError: We choose to put points at the positive end of x-dir into the last box, not the first box.
            ix = ix - 1
        if iy == N_y:
            iy = iy - 1
        if iz == N_z:
            iz = iz - 1
        if ix == -1:   # To avoid IndexError: This does not bother us since round(-0.5)=0.0, but in case box is not [0, max]^3...
            ix = ix + 1
        if iy == -1:
            iy = iy + 1
        if iz == -1:
            iz = iz + 1
        nx = xx_mem[ix,iy,iz]
        ny = yy_mem[ix,iy,iz]
        nz = zz_mem[ix,iy,iz]
        
        # Distance from particle to nearest cell's right edge
        dx = x[p] - nx
        dy = y[p] - ny
        dz = z[p] - nz
        
        # Note that mod N deals with PBC
        facx = 1
        facy = 1 
        facz = 1
        if dx < - hx:
            facx = -1
        if dy < - hy:
            facy = -1
        if dz < - hz:
            facz = -1
        ix1 = int(cyround((ix + facx) % N_x))
        iy1 = int(cyround((iy + facy) % N_y))
        iz1 = int(cyround((iz + facz) % N_z))
        if dx < - hx:
            dlx = (2*hx + dx)*(-1)
        else:
            dlx = dx
        if dy < - hy:
            dly = (2*hy + dy)*(-1)
        else:
            dly = dy
        if dz < - hz:
            dlz = (2*hz + dz)*(-1)
        else:
            dlz = dz
            
        # Place mass into the 8 neighboring cells with appropriate weights
        rhoCIC[openmp.omp_get_thread_num(), ix,iy,iz]    += masses[p] * (hx-dlx) * (hy-dly) * (hz-dlz) / (volx*voly*volz)**(2/3);
        rhoCIC[openmp.omp_get_thread_num(), ix,iy1,iz]   += masses[p] * (hx+dlx) * (hy-dly) * (hz-dlz) / (volx*voly*volz)**(2/3);
        rhoCIC[openmp.omp_get_thread_num(), ix1,iy,iz]   += masses[p] * (hx-dlx) * (hy+dly) * (hz-dlz) / (volx*voly*volz)**(2/3);
        rhoCIC[openmp.omp_get_thread_num(), ix,iy,iz1]   += masses[p] * (hx-dlx) * (hy-dly) * (hz+dlz) / (volx*voly*volz)**(2/3);
        rhoCIC[openmp.omp_get_thread_num(), ix1,iy1,iz]  += masses[p] * (hx+dlx) * (hy+dly) * (hz-dlz) / (volx*voly*volz)**(2/3);
        rhoCIC[openmp.omp_get_thread_num(), ix1,iy,iz1]  += masses[p] * (hx-dlx) * (hy+dly) * (hz+dlz) / (volx*voly*volz)**(2/3);
        rhoCIC[openmp.omp_get_thread_num(), ix,iy1,iz1]  += masses[p] * (hx+dlx) * (hy-dly) * (hz+dlz) / (volx*voly*volz)**(2/3);
        rhoCIC[openmp.omp_get_thread_num(), ix1,iy1,iz1] += masses[p] * (hx+dlx) * (hy+dly) * (hz+dlz) / (volx*voly*volz)**(2/3);
    rho = np.zeros((N_x, N_y, N_z), dtype = np.float32)
    for i in range(openmp.omp_get_max_threads()):
        rho += rhoCIC[i]
    return rho # Has (N_x, N_y, N_z)-kind of ordering. No transpose needed.


def inverseCICTidalPBC(float[:] x, float[:] y, float[:] z, float[:,:,:,:,:] tidal, float L_BOX, int N, traceless = False, normalize = False):
    """
    Infer tidal field at halo positions via inverse CIC
    Periodic boundary conditions (PBC) are assumed.
    Parameters
    ----------
    x, y, z : 1D arrays, halo positions
    tidal: (3,3,N,N,N)-shaped array, contains tidal field at every grid point
    L_BOX: box-size
    N: box resolution
    Returns
    ----------
    A (nb_halos,3,3)-shaped float array
    The tidal field locally at the nb_halos halos
    """
    
    # Assert correct shape
    assert tidal.shape[0] == 3 and tidal.shape[1] == 3 and tidal.shape[2] == N and tidal.shape[3] == N and tidal.shape[4] == N, \
        'Shape of tidal array is wrong.'
    
    # Grid parameters
    cdef float delx = L_BOX/N
    cdef float vol = delx**3
    cdef float hx = delx / 2
    cdef float[:] xlin = ((np.arange(1,N+1))*delx).astype('float32')  # The cells' right edges
    [xx, yy, zz] = np.meshgrid(xlin, xlin, xlin, indexing='ij') 
    
    # CIC binnning
    cdef int ix
    cdef int iy
    cdef int iz
    cdef float[:,:,:] xx_mem = xx.astype(np.float32)
    cdef float[:,:,:] yy_mem = yy.astype(np.float32)
    cdef float[:,:,:] zz_mem = zz.astype(np.float32)
    cdef float dx
    cdef float dy
    cdef float dz
    cdef float nx
    cdef float ny
    cdef float nz
    cdef int facx
    cdef int facy
    cdef int facz
    cdef int ix1
    cdef int iy1
    cdef int iz1
    cdef float dlx
    cdef float dly
    cdef float dlz
    cdef Py_ssize_t p
    cdef int len_x = len(x)
    cdef float[:,:] tidal_xx = np.zeros((openmp.omp_get_max_threads(),len_x), dtype=np.float32)
    cdef float[:,:] tidal_xy = np.zeros((openmp.omp_get_max_threads(),len_x), dtype=np.float32)
    cdef float[:,:] tidal_xz = np.zeros((openmp.omp_get_max_threads(),len_x), dtype=np.float32)
    cdef float[:,:] tidal_yy = np.zeros((openmp.omp_get_max_threads(),len_x), dtype=np.float32)
    cdef float[:,:] tidal_yz = np.zeros((openmp.omp_get_max_threads(),len_x), dtype=np.float32)
    cdef float[:,:] tidal_zz = np.zeros((openmp.omp_get_max_threads(),len_x), dtype=np.float32)
    
    for p in prange(len_x, nogil = True):
        # Find nearest cell
        ix = int(cyround(x[p] / L_BOX * N - 0.5)) 
        iy = int(cyround(y[p] / L_BOX * N - 0.5)) 
        iz = int(cyround(z[p] / L_BOX * N - 0.5)) 
        if ix == N:    # To avoid IndexError: We choose to put points at the positive end of x-dir into the last box, not the first box.
            ix = ix - 1
        if iy == N:
            iy = iy - 1
        if iz == N:
            iz = iz - 1
        if ix == -1:   # To avoid IndexError: This does not bother us since round(-0.5)=0.0, but in case box is not [0, max]^3...
            ix = ix + 1
        if iy == -1:
            iy = iy + 1
        if iz == -1:
            iz = iz + 1
        nx = xx_mem[ix,iy,iz]
        ny = yy_mem[ix,iy,iz]
        nz = zz_mem[ix,iy,iz]
        
        # Distance from particle to nearest cell's right edge
        dx = x[p] - nx
        dy = y[p] - ny
        dz = z[p] - nz
        
        # Note that mod N deals with PBC
        facx = 1
        facy = 1 
        facz = 1
        if dx < - hx:
            facx = -1
        if dy < - hx:
            facy = -1
        if dz < - hx:
            facz = -1
        ix1 = int(cyround((ix + facx) % N))
        iy1 = int(cyround((iy + facy) % N))
        iz1 = int(cyround((iz + facz) % N))
        if dx < - hx:
            dlx = (2*hx + dx)*(-1)
        else:
            dlx = dx
        if dy < - hx:
            dly = (2*hx + dy)*(-1)
        else:
            dly = dy
        if dz < - hx:
            dlz = (2*hx + dz)*(-1)
        else:
            dlz = dz
        
        # Place mass into the 8 neighboring cells with appropriate weights
        tidal_xx[openmp.omp_get_thread_num(), p] += (hx-dlx) * (hx-dly) * (hx-dlz) * tidal[0,0,ix,iy,iz]   / vol;
        tidal_xx[openmp.omp_get_thread_num(), p] += (hx+dlx) * (hx-dly) * (hx-dlz) * tidal[0,0,ix,iy1,iz]  / vol;
        tidal_xx[openmp.omp_get_thread_num(), p] += (hx-dlx) * (hx+dly) * (hx-dlz) * tidal[0,0,ix1,iy,iz]  / vol;
        tidal_xx[openmp.omp_get_thread_num(), p] += (hx-dlx) * (hx-dly) * (hx+dlz) * tidal[0,0,ix,iy,iz1]  / vol;
        tidal_xx[openmp.omp_get_thread_num(), p] += (hx+dlx) * (hx+dly) * (hx-dlz) * tidal[0,0,ix1,iy1,iz] / vol;
        tidal_xx[openmp.omp_get_thread_num(), p] += (hx-dlx) * (hx+dly) * (hx+dlz) * tidal[0,0,ix1,iy,iz1] / vol;
        tidal_xx[openmp.omp_get_thread_num(), p] += (hx+dlx) * (hx-dly) * (hx+dlz) * tidal[0,0,ix,iy1,iz1] / vol;
        tidal_xx[openmp.omp_get_thread_num(), p] += (hx+dlx) * (hx+dly) * (hx+dlz) * tidal[0,0,ix1,iy1,iz1]/ vol;
        
        # Place mass into the 8 neighboring cells with appropriate weights
        tidal_xy[openmp.omp_get_thread_num(), p] += (hx-dlx) * (hx-dly) * (hx-dlz) * tidal[0,1,ix,iy,iz]   / vol;
        tidal_xy[openmp.omp_get_thread_num(), p] += (hx+dlx) * (hx-dly) * (hx-dlz) * tidal[0,1,ix,iy1,iz]  / vol;
        tidal_xy[openmp.omp_get_thread_num(), p] += (hx-dlx) * (hx+dly) * (hx-dlz) * tidal[0,1,ix1,iy,iz]  / vol;
        tidal_xy[openmp.omp_get_thread_num(), p] += (hx-dlx) * (hx-dly) * (hx+dlz) * tidal[0,1,ix,iy,iz1]  / vol;
        tidal_xy[openmp.omp_get_thread_num(), p] += (hx+dlx) * (hx+dly) * (hx-dlz) * tidal[0,1,ix1,iy1,iz] / vol;
        tidal_xy[openmp.omp_get_thread_num(), p] += (hx-dlx) * (hx+dly) * (hx+dlz) * tidal[0,1,ix1,iy,iz1] / vol;
        tidal_xy[openmp.omp_get_thread_num(), p] += (hx+dlx) * (hx-dly) * (hx+dlz) * tidal[0,1,ix,iy1,iz1] / vol;
        tidal_xy[openmp.omp_get_thread_num(), p] += (hx+dlx) * (hx+dly) * (hx+dlz) * tidal[0,1,ix1,iy1,iz1]/ vol;
        
        # Place mass into the 8 neighboring cells with appropriate weights
        tidal_xz[openmp.omp_get_thread_num(), p] += (hx-dlx) * (hx-dly) * (hx-dlz) * tidal[0,2,ix,iy,iz]   / vol;
        tidal_xz[openmp.omp_get_thread_num(), p] += (hx+dlx) * (hx-dly) * (hx-dlz) * tidal[0,2,ix,iy1,iz]  / vol;
        tidal_xz[openmp.omp_get_thread_num(), p] += (hx-dlx) * (hx+dly) * (hx-dlz) * tidal[0,2,ix1,iy,iz]  / vol;
        tidal_xz[openmp.omp_get_thread_num(), p] += (hx-dlx) * (hx-dly) * (hx+dlz) * tidal[0,2,ix,iy,iz1]  / vol;
        tidal_xz[openmp.omp_get_thread_num(), p] += (hx+dlx) * (hx+dly) * (hx-dlz) * tidal[0,2,ix1,iy1,iz] / vol;
        tidal_xz[openmp.omp_get_thread_num(), p] += (hx-dlx) * (hx+dly) * (hx+dlz) * tidal[0,2,ix1,iy,iz1] / vol;
        tidal_xz[openmp.omp_get_thread_num(), p] += (hx+dlx) * (hx-dly) * (hx+dlz) * tidal[0,2,ix,iy1,iz1] / vol;
        tidal_xz[openmp.omp_get_thread_num(), p] += (hx+dlx) * (hx+dly) * (hx+dlz) * tidal[0,2,ix1,iy1,iz1]/ vol;
        
        # Place mass into the 8 neighboring cells with appropriate weights
        tidal_yy[openmp.omp_get_thread_num(), p] += (hx-dlx) * (hx-dly) * (hx-dlz) * tidal[1,1,ix,iy,iz]   / vol;
        tidal_yy[openmp.omp_get_thread_num(), p] += (hx+dlx) * (hx-dly) * (hx-dlz) * tidal[1,1,ix,iy1,iz]  / vol;
        tidal_yy[openmp.omp_get_thread_num(), p] += (hx-dlx) * (hx+dly) * (hx-dlz) * tidal[1,1,ix1,iy,iz]  / vol;
        tidal_yy[openmp.omp_get_thread_num(), p] += (hx-dlx) * (hx-dly) * (hx+dlz) * tidal[1,1,ix,iy,iz1]  / vol;
        tidal_yy[openmp.omp_get_thread_num(), p] += (hx+dlx) * (hx+dly) * (hx-dlz) * tidal[1,1,ix1,iy1,iz] / vol;
        tidal_yy[openmp.omp_get_thread_num(), p] += (hx-dlx) * (hx+dly) * (hx+dlz) * tidal[1,1,ix1,iy,iz1] / vol;
        tidal_yy[openmp.omp_get_thread_num(), p] += (hx+dlx) * (hx-dly) * (hx+dlz) * tidal[1,1,ix,iy1,iz1] / vol;
        tidal_yy[openmp.omp_get_thread_num(), p] += (hx+dlx) * (hx+dly) * (hx+dlz) * tidal[1,1,ix1,iy1,iz1]/ vol;
        
        # Place mass into the 8 neighboring cells with appropriate weights
        tidal_yz[openmp.omp_get_thread_num(), p] += (hx-dlx) * (hx-dly) * (hx-dlz) * tidal[1,2,ix,iy,iz]   / vol;
        tidal_yz[openmp.omp_get_thread_num(), p] += (hx+dlx) * (hx-dly) * (hx-dlz) * tidal[1,2,ix,iy1,iz]  / vol;
        tidal_yz[openmp.omp_get_thread_num(), p] += (hx-dlx) * (hx+dly) * (hx-dlz) * tidal[1,2,ix1,iy,iz]  / vol;
        tidal_yz[openmp.omp_get_thread_num(), p] += (hx-dlx) * (hx-dly) * (hx+dlz) * tidal[1,2,ix,iy,iz1]  / vol;
        tidal_yz[openmp.omp_get_thread_num(), p] += (hx+dlx) * (hx+dly) * (hx-dlz) * tidal[1,2,ix1,iy1,iz] / vol;
        tidal_yz[openmp.omp_get_thread_num(), p] += (hx-dlx) * (hx+dly) * (hx+dlz) * tidal[1,2,ix1,iy,iz1] / vol;
        tidal_yz[openmp.omp_get_thread_num(), p] += (hx+dlx) * (hx-dly) * (hx+dlz) * tidal[1,2,ix,iy1,iz1] / vol;
        tidal_yz[openmp.omp_get_thread_num(), p] += (hx+dlx) * (hx+dly) * (hx+dlz) * tidal[1,2,ix1,iy1,iz1]/ vol;
        
        # Place mass into the 8 neighboring cells with appropriate weights
        tidal_zz[openmp.omp_get_thread_num(), p] += (hx-dlx) * (hx-dly) * (hx-dlz) * tidal[2,2,ix,iy,iz]   / vol;
        tidal_zz[openmp.omp_get_thread_num(), p] += (hx+dlx) * (hx-dly) * (hx-dlz) * tidal[2,2,ix,iy1,iz]  / vol;
        tidal_zz[openmp.omp_get_thread_num(), p] += (hx-dlx) * (hx+dly) * (hx-dlz) * tidal[2,2,ix1,iy,iz]  / vol;
        tidal_zz[openmp.omp_get_thread_num(), p] += (hx-dlx) * (hx-dly) * (hx+dlz) * tidal[2,2,ix,iy,iz1]  / vol;
        tidal_zz[openmp.omp_get_thread_num(), p] += (hx+dlx) * (hx+dly) * (hx-dlz) * tidal[2,2,ix1,iy1,iz] / vol;
        tidal_zz[openmp.omp_get_thread_num(), p] += (hx-dlx) * (hx+dly) * (hx+dlz) * tidal[2,2,ix1,iy,iz1] / vol;
        tidal_zz[openmp.omp_get_thread_num(), p] += (hx+dlx) * (hx-dly) * (hx+dlz) * tidal[2,2,ix,iy1,iz1] / vol;
        tidal_zz[openmp.omp_get_thread_num(), p] += (hx+dlx) * (hx+dly) * (hx+dlz) * tidal[2,2,ix1,iy1,iz1]/ vol;
        
    tidal_xx_all = np.zeros((len_x), dtype = np.float32)
    for i in range(openmp.omp_get_max_threads()):
        tidal_xx_all += tidal_xx[i]
    tidal_xy_all = np.zeros((len_x), dtype = np.float32)
    for i in range(openmp.omp_get_max_threads()):
        tidal_xy_all += tidal_xy[i]
    tidal_xz_all = np.zeros((len_x), dtype = np.float32)
    for i in range(openmp.omp_get_max_threads()):
        tidal_xz_all += tidal_xz[i]
    tidal_yy_all = np.zeros((len_x), dtype = np.float32)
    for i in range(openmp.omp_get_max_threads()):
        tidal_yy_all += tidal_yy[i]
    tidal_yz_all = np.zeros((len_x), dtype = np.float32)
    for i in range(openmp.omp_get_max_threads()):
        tidal_yz_all += tidal_yz[i]
    tidal_zz_all = np.zeros((len_x), dtype = np.float32)
    for i in range(openmp.omp_get_max_threads()):
        tidal_zz_all += tidal_zz[i]
        
    if traceless == True:
        tidal_tmp_xx = deepcopy(tidal_xx_all)
        tidal_tmp_yy = deepcopy(tidal_yy_all)
        tidal_tmp_zz = deepcopy(tidal_zz_all)
        tidal_xx_all -= 1/3*(tidal_tmp_xx+tidal_tmp_yy+tidal_tmp_zz)
        tidal_yy_all -= 1/3*(tidal_tmp_xx+tidal_tmp_yy+tidal_tmp_zz)
        tidal_zz_all -= 1/3*(tidal_tmp_xx+tidal_tmp_yy+tidal_tmp_zz)

    if normalize == True:
        tidal_tmp_xx = deepcopy(tidal_xx_all)
        tidal_tmp_xy = deepcopy(tidal_xy_all)
        tidal_tmp_xz = deepcopy(tidal_xz_all)
        tidal_tmp_yy = deepcopy(tidal_yy_all)
        tidal_tmp_yz = deepcopy(tidal_yz_all)
        tidal_tmp_zz = deepcopy(tidal_zz_all)
        tidal_xx_all /= (tidal_tmp_xx**2 + tidal_tmp_yy**2 + tidal_tmp_zz**2 + 2*tidal_tmp_xy**2 + 2*tidal_tmp_xz**2 + 2*tidal_tmp_yz**2)**(1/2)
        tidal_xy_all /= (tidal_tmp_xx**2 + tidal_tmp_yy**2 + tidal_tmp_zz**2 + 2*tidal_tmp_xy**2 + 2*tidal_tmp_xz**2 + 2*tidal_tmp_yz**2)**(1/2)
        tidal_xz_all /= (tidal_tmp_xx**2 + tidal_tmp_yy**2 + tidal_tmp_zz**2 + 2*tidal_tmp_xy**2 + 2*tidal_tmp_xz**2 + 2*tidal_tmp_yz**2)**(1/2)
        tidal_yy_all /= (tidal_tmp_xx**2 + tidal_tmp_yy**2 + tidal_tmp_zz**2 + 2*tidal_tmp_xy**2 + 2*tidal_tmp_xz**2 + 2*tidal_tmp_yz**2)**(1/2)
        tidal_yz_all /= (tidal_tmp_xx**2 + tidal_tmp_yy**2 + tidal_tmp_zz**2 + 2*tidal_tmp_xy**2 + 2*tidal_tmp_xz**2 + 2*tidal_tmp_yz**2)**(1/2)
        tidal_zz_all /= (tidal_tmp_xx**2 + tidal_tmp_yy**2 + tidal_tmp_zz**2 + 2*tidal_tmp_xy**2 + 2*tidal_tmp_xz**2 + 2*tidal_tmp_yz**2)**(1/2)
    
    return tidal_xx_all, tidal_xy_all, tidal_xz_all, tidal_yy_all, tidal_yz_all

def inverseCICTidalPBCUnequal(float[:] x, float[:] y, float[:] z, float[:,:,:,:,:] tidal, float L_x, float L_y, float L_z, int N_x, int N_y, int N_z, traceless = False, normalize = False):
    """
    Infer tidal field at halo positions via inverse CIC
    Periodic boundary conditions (PBC) are assumed.
    Parameters
    ----------
    x, y, z : 1D arrays, halo positions
    tidal: (3,3,N,N,N)-shaped array, contains tidal field at every grid point
    L_BOX: box-size
    N: box resolution
    Returns
    ----------
    A (nb_halos,3,3)-shaped float array
    The tidal field locally at the nb_halos halos
    """
    
    # Assert correct shape
    assert tidal.shape[0] == 3 and tidal.shape[1] == 3 and tidal.shape[2] == N_x and tidal.shape[3] == N_y and tidal.shape[4] == N_z, \
        'Shape of tidal array is wrong.'
    
    # Grid parameters
    cdef float delx = L_x/N_x
    cdef float dely = L_y/N_y
    cdef float delz = L_z/N_z
    cdef float volx = delx**3
    cdef float voly = dely**3
    cdef float volz = delz**3
    cdef float hx = delx / 2
    cdef float hy = dely / 2
    cdef float hz = delz / 2
    cdef float[:] xlin = ((np.arange(1,N_x+1))*delx).astype('float32')  # The cells' right edges
    cdef float[:] ylin = ((np.arange(1,N_y+1))*dely).astype('float32')  # The cells' right edges
    cdef float[:] zlin = ((np.arange(1,N_z+1))*delz).astype('float32')  # The cells' right edges
    [xx, yy, zz] = np.meshgrid(xlin, ylin, zlin, indexing='ij') 
    
    # CIC binnning
    cdef int ix
    cdef int iy
    cdef int iz
    cdef float[:,:,:] xx_mem = xx.astype(np.float32)
    cdef float[:,:,:] yy_mem = yy.astype(np.float32)
    cdef float[:,:,:] zz_mem = zz.astype(np.float32)
    cdef float dx
    cdef float dy
    cdef float dz
    cdef float nx
    cdef float ny
    cdef float nz
    cdef int facx
    cdef int facy
    cdef int facz
    cdef int ix1
    cdef int iy1
    cdef int iz1
    cdef float dlx
    cdef float dly
    cdef float dlz
    cdef Py_ssize_t p
    cdef int len_x = len(x)
    cdef float[:,:] tidal_xx = np.zeros((openmp.omp_get_max_threads(),len_x), dtype=np.float32)
    cdef float[:,:] tidal_xy = np.zeros((openmp.omp_get_max_threads(),len_x), dtype=np.float32)
    cdef float[:,:] tidal_xz = np.zeros((openmp.omp_get_max_threads(),len_x), dtype=np.float32)
    cdef float[:,:] tidal_yy = np.zeros((openmp.omp_get_max_threads(),len_x), dtype=np.float32)
    cdef float[:,:] tidal_yz = np.zeros((openmp.omp_get_max_threads(),len_x), dtype=np.float32)
    cdef float[:,:] tidal_zz = np.zeros((openmp.omp_get_max_threads(),len_x), dtype=np.float32)
    
    for p in prange(len_x, nogil = True):
        # Find nearest cell
        ix = int(cyround(x[p] / L_x * N_x - 0.5)) 
        iy = int(cyround(y[p] / L_y * N_y - 0.5)) 
        iz = int(cyround(z[p] / L_z * N_z - 0.5)) 
        if ix == N_x:    # To avoid IndexError: We choose to put points at the positive end of x-dir into the last box, not the first box.
            ix = ix - 1
        if iy == N_y:
            iy = iy - 1
        if iz == N_z:
            iz = iz - 1
        if ix == -1:   # To avoid IndexError: This does not bother us since round(-0.5)=0.0, but in case box is not [0, max]^3...
            ix = ix + 1
        if iy == -1:
            iy = iy + 1
        if iz == -1:
            iz = iz + 1
        nx = xx_mem[ix,iy,iz]
        ny = yy_mem[ix,iy,iz]
        nz = zz_mem[ix,iy,iz]
        
        # Distance from particle to nearest cell's right edge
        dx = x[p] - nx
        dy = y[p] - ny
        dz = z[p] - nz
        
        # Note that mod N deals with PBC
        facx = 1
        facy = 1 
        facz = 1
        if dx < - hx:
            facx = -1
        if dy < - hy:
            facy = -1
        if dz < - hz:
            facz = -1
        ix1 = int(cyround((ix + facx) % N_x))
        iy1 = int(cyround((iy + facy) % N_y))
        iz1 = int(cyround((iz + facz) % N_z))
        if dx < - hx:
            dlx = (2*hx + dx)*(-1)
        else:
            dlx = dx
        if dy < - hy:
            dly = (2*hy + dy)*(-1)
        else:
            dly = dy
        if dz < - hz:
            dlz = (2*hz + dz)*(-1)
        else:
            dlz = dz
        
        # Place mass into the 8 neighboring cells with appropriate weights
        tidal_xx[openmp.omp_get_thread_num(), p] += (hx-dlx) * (hy-dly) * (hz-dlz) * tidal[0,0,ix,iy,iz]   / (volx*voly*volz)**(1/3)
        tidal_xx[openmp.omp_get_thread_num(), p] += (hx+dlx) * (hy-dly) * (hz-dlz) * tidal[0,0,ix,iy1,iz]  / (volx*voly*volz)**(1/3)
        tidal_xx[openmp.omp_get_thread_num(), p] += (hx-dlx) * (hy+dly) * (hz-dlz) * tidal[0,0,ix1,iy,iz]  / (volx*voly*volz)**(1/3)
        tidal_xx[openmp.omp_get_thread_num(), p] += (hx-dlx) * (hy-dly) * (hz+dlz) * tidal[0,0,ix,iy,iz1]  / (volx*voly*volz)**(1/3)
        tidal_xx[openmp.omp_get_thread_num(), p] += (hx+dlx) * (hy+dly) * (hz-dlz) * tidal[0,0,ix1,iy1,iz] / (volx*voly*volz)**(1/3)
        tidal_xx[openmp.omp_get_thread_num(), p] += (hx-dlx) * (hy+dly) * (hz+dlz) * tidal[0,0,ix1,iy,iz1] / (volx*voly*volz)**(1/3)
        tidal_xx[openmp.omp_get_thread_num(), p] += (hx+dlx) * (hy-dly) * (hz+dlz) * tidal[0,0,ix,iy1,iz1] / (volx*voly*volz)**(1/3)
        tidal_xx[openmp.omp_get_thread_num(), p] += (hx+dlx) * (hy+dly) * (hz+dlz) * tidal[0,0,ix1,iy1,iz1]/ (volx*voly*volz)**(1/3)
        
        # Place mass into the 8 neighboring cells with appropriate weights
        tidal_xy[openmp.omp_get_thread_num(), p] += (hx-dlx) * (hy-dly) * (hz-dlz) * tidal[0,1,ix,iy,iz]   / (volx*voly*volz)**(1/3)
        tidal_xy[openmp.omp_get_thread_num(), p] += (hx+dlx) * (hy-dly) * (hz-dlz) * tidal[0,1,ix,iy1,iz]  / (volx*voly*volz)**(1/3)
        tidal_xy[openmp.omp_get_thread_num(), p] += (hx-dlx) * (hy+dly) * (hz-dlz) * tidal[0,1,ix1,iy,iz]  / (volx*voly*volz)**(1/3)
        tidal_xy[openmp.omp_get_thread_num(), p] += (hx-dlx) * (hy-dly) * (hz+dlz) * tidal[0,1,ix,iy,iz1]  / (volx*voly*volz)**(1/3)
        tidal_xy[openmp.omp_get_thread_num(), p] += (hx+dlx) * (hy+dly) * (hz-dlz) * tidal[0,1,ix1,iy1,iz] / (volx*voly*volz)**(1/3)
        tidal_xy[openmp.omp_get_thread_num(), p] += (hx-dlx) * (hy+dly) * (hz+dlz) * tidal[0,1,ix1,iy,iz1] / (volx*voly*volz)**(1/3)
        tidal_xy[openmp.omp_get_thread_num(), p] += (hx+dlx) * (hy-dly) * (hz+dlz) * tidal[0,1,ix,iy1,iz1] / (volx*voly*volz)**(1/3)
        tidal_xy[openmp.omp_get_thread_num(), p] += (hx+dlx) * (hy+dly) * (hz+dlz) * tidal[0,1,ix1,iy1,iz1]/ (volx*voly*volz)**(1/3)
        
        # Place mass into the 8 neighboring cells with appropriate weights
        tidal_xz[openmp.omp_get_thread_num(), p] += (hx-dlx) * (hy-dly) * (hz-dlz) * tidal[0,2,ix,iy,iz]   / (volx*voly*volz)**(1/3)
        tidal_xz[openmp.omp_get_thread_num(), p] += (hx+dlx) * (hy-dly) * (hz-dlz) * tidal[0,2,ix,iy1,iz]  / (volx*voly*volz)**(1/3)
        tidal_xz[openmp.omp_get_thread_num(), p] += (hx-dlx) * (hy+dly) * (hz-dlz) * tidal[0,2,ix1,iy,iz]  / (volx*voly*volz)**(1/3)
        tidal_xz[openmp.omp_get_thread_num(), p] += (hx-dlx) * (hy-dly) * (hz+dlz) * tidal[0,2,ix,iy,iz1]  / (volx*voly*volz)**(1/3)
        tidal_xz[openmp.omp_get_thread_num(), p] += (hx+dlx) * (hy+dly) * (hz-dlz) * tidal[0,2,ix1,iy1,iz] / (volx*voly*volz)**(1/3)
        tidal_xz[openmp.omp_get_thread_num(), p] += (hx-dlx) * (hy+dly) * (hz+dlz) * tidal[0,2,ix1,iy,iz1] / (volx*voly*volz)**(1/3)
        tidal_xz[openmp.omp_get_thread_num(), p] += (hx+dlx) * (hy-dly) * (hz+dlz) * tidal[0,2,ix,iy1,iz1] / (volx*voly*volz)**(1/3)
        tidal_xz[openmp.omp_get_thread_num(), p] += (hx+dlx) * (hy+dly) * (hz+dlz) * tidal[0,2,ix1,iy1,iz1]/ (volx*voly*volz)**(1/3)
        
        # Place mass into the 8 neighboring cells with appropriate weights
        tidal_yy[openmp.omp_get_thread_num(), p] += (hx-dlx) * (hy-dly) * (hz-dlz) * tidal[1,1,ix,iy,iz]   / (volx*voly*volz)**(1/3)
        tidal_yy[openmp.omp_get_thread_num(), p] += (hx+dlx) * (hy-dly) * (hz-dlz) * tidal[1,1,ix,iy1,iz]  / (volx*voly*volz)**(1/3)
        tidal_yy[openmp.omp_get_thread_num(), p] += (hx-dlx) * (hy+dly) * (hz-dlz) * tidal[1,1,ix1,iy,iz]  / (volx*voly*volz)**(1/3)
        tidal_yy[openmp.omp_get_thread_num(), p] += (hx-dlx) * (hy-dly) * (hz+dlz) * tidal[1,1,ix,iy,iz1]  / (volx*voly*volz)**(1/3)
        tidal_yy[openmp.omp_get_thread_num(), p] += (hx+dlx) * (hy+dly) * (hz-dlz) * tidal[1,1,ix1,iy1,iz] / (volx*voly*volz)**(1/3)
        tidal_yy[openmp.omp_get_thread_num(), p] += (hx-dlx) * (hy+dly) * (hz+dlz) * tidal[1,1,ix1,iy,iz1] / (volx*voly*volz)**(1/3)
        tidal_yy[openmp.omp_get_thread_num(), p] += (hx+dlx) * (hy-dly) * (hz+dlz) * tidal[1,1,ix,iy1,iz1] / (volx*voly*volz)**(1/3)
        tidal_yy[openmp.omp_get_thread_num(), p] += (hx+dlx) * (hy+dly) * (hz+dlz) * tidal[1,1,ix1,iy1,iz1]/ (volx*voly*volz)**(1/3)
        
        # Place mass into the 8 neighboring cells with appropriate weights
        tidal_yz[openmp.omp_get_thread_num(), p] += (hx-dlx) * (hy-dly) * (hz-dlz) * tidal[1,2,ix,iy,iz]   / (volx*voly*volz)**(1/3)
        tidal_yz[openmp.omp_get_thread_num(), p] += (hx+dlx) * (hy-dly) * (hz-dlz) * tidal[1,2,ix,iy1,iz]  / (volx*voly*volz)**(1/3)
        tidal_yz[openmp.omp_get_thread_num(), p] += (hx-dlx) * (hy+dly) * (hz-dlz) * tidal[1,2,ix1,iy,iz]  / (volx*voly*volz)**(1/3)
        tidal_yz[openmp.omp_get_thread_num(), p] += (hx-dlx) * (hy-dly) * (hz+dlz) * tidal[1,2,ix,iy,iz1]  / (volx*voly*volz)**(1/3)
        tidal_yz[openmp.omp_get_thread_num(), p] += (hx+dlx) * (hy+dly) * (hz-dlz) * tidal[1,2,ix1,iy1,iz] / (volx*voly*volz)**(1/3)
        tidal_yz[openmp.omp_get_thread_num(), p] += (hx-dlx) * (hy+dly) * (hz+dlz) * tidal[1,2,ix1,iy,iz1] / (volx*voly*volz)**(1/3)
        tidal_yz[openmp.omp_get_thread_num(), p] += (hx+dlx) * (hy-dly) * (hz+dlz) * tidal[1,2,ix,iy1,iz1] / (volx*voly*volz)**(1/3)
        tidal_yz[openmp.omp_get_thread_num(), p] += (hx+dlx) * (hy+dly) * (hz+dlz) * tidal[1,2,ix1,iy1,iz1]/ (volx*voly*volz)**(1/3)
        
        # Place mass into the 8 neighboring cells with appropriate weights
        tidal_zz[openmp.omp_get_thread_num(), p] += (hx-dlx) * (hy-dly) * (hz-dlz) * tidal[2,2,ix,iy,iz]   / (volx*voly*volz)**(1/3)
        tidal_zz[openmp.omp_get_thread_num(), p] += (hx+dlx) * (hy-dly) * (hz-dlz) * tidal[2,2,ix,iy1,iz]  / (volx*voly*volz)**(1/3)
        tidal_zz[openmp.omp_get_thread_num(), p] += (hx-dlx) * (hy+dly) * (hz-dlz) * tidal[2,2,ix1,iy,iz]  / (volx*voly*volz)**(1/3)
        tidal_zz[openmp.omp_get_thread_num(), p] += (hx-dlx) * (hy-dly) * (hz+dlz) * tidal[2,2,ix,iy,iz1]  / (volx*voly*volz)**(1/3)
        tidal_zz[openmp.omp_get_thread_num(), p] += (hx+dlx) * (hy+dly) * (hz-dlz) * tidal[2,2,ix1,iy1,iz] / (volx*voly*volz)**(1/3)
        tidal_zz[openmp.omp_get_thread_num(), p] += (hx-dlx) * (hy+dly) * (hz+dlz) * tidal[2,2,ix1,iy,iz1] / (volx*voly*volz)**(1/3)
        tidal_zz[openmp.omp_get_thread_num(), p] += (hx+dlx) * (hy-dly) * (hz+dlz) * tidal[2,2,ix,iy1,iz1] / (volx*voly*volz)**(1/3)
        tidal_zz[openmp.omp_get_thread_num(), p] += (hx+dlx) * (hy+dly) * (hz+dlz) * tidal[2,2,ix1,iy1,iz1]/ (volx*voly*volz)**(1/3)
        
    tidal_xx_all = np.zeros((len_x), dtype = np.float32)
    for i in range(openmp.omp_get_max_threads()):
        tidal_xx_all += tidal_xx[i]
    tidal_xy_all = np.zeros((len_x), dtype = np.float32)
    for i in range(openmp.omp_get_max_threads()):
        tidal_xy_all += tidal_xy[i]
    tidal_xz_all = np.zeros((len_x), dtype = np.float32)
    for i in range(openmp.omp_get_max_threads()):
        tidal_xz_all += tidal_xz[i]
    tidal_yy_all = np.zeros((len_x), dtype = np.float32)
    for i in range(openmp.omp_get_max_threads()):
        tidal_yy_all += tidal_yy[i]
    tidal_yz_all = np.zeros((len_x), dtype = np.float32)
    for i in range(openmp.omp_get_max_threads()):
        tidal_yz_all += tidal_yz[i]
    tidal_zz_all = np.zeros((len_x), dtype = np.float32)
    for i in range(openmp.omp_get_max_threads()):
        tidal_zz_all += tidal_zz[i]
        
    if traceless == True:
        tidal_tmp_xx = deepcopy(tidal_xx_all)
        tidal_tmp_yy = deepcopy(tidal_yy_all)
        tidal_tmp_zz = deepcopy(tidal_zz_all)
        tidal_xx_all -= 1/3*(tidal_tmp_xx+tidal_tmp_yy+tidal_tmp_zz)
        tidal_yy_all -= 1/3*(tidal_tmp_xx+tidal_tmp_yy+tidal_tmp_zz)
        tidal_zz_all -= 1/3*(tidal_tmp_xx+tidal_tmp_yy+tidal_tmp_zz)

    if normalize == True:
        tidal_tmp_xx = deepcopy(tidal_xx_all)
        tidal_tmp_xy = deepcopy(tidal_xy_all)
        tidal_tmp_xz = deepcopy(tidal_xz_all)
        tidal_tmp_yy = deepcopy(tidal_yy_all)
        tidal_tmp_yz = deepcopy(tidal_yz_all)
        tidal_tmp_zz = deepcopy(tidal_zz_all)
        tidal_xx_all /= (tidal_tmp_xx**2 + tidal_tmp_yy**2 + tidal_tmp_zz**2 + 2*tidal_tmp_xy**2 + 2*tidal_tmp_xz**2 + 2*tidal_tmp_yz**2)**(1/2)
        tidal_xy_all /= (tidal_tmp_xx**2 + tidal_tmp_yy**2 + tidal_tmp_zz**2 + 2*tidal_tmp_xy**2 + 2*tidal_tmp_xz**2 + 2*tidal_tmp_yz**2)**(1/2)
        tidal_xz_all /= (tidal_tmp_xx**2 + tidal_tmp_yy**2 + tidal_tmp_zz**2 + 2*tidal_tmp_xy**2 + 2*tidal_tmp_xz**2 + 2*tidal_tmp_yz**2)**(1/2)
        tidal_yy_all /= (tidal_tmp_xx**2 + tidal_tmp_yy**2 + tidal_tmp_zz**2 + 2*tidal_tmp_xy**2 + 2*tidal_tmp_xz**2 + 2*tidal_tmp_yz**2)**(1/2)
        tidal_yz_all /= (tidal_tmp_xx**2 + tidal_tmp_yy**2 + tidal_tmp_zz**2 + 2*tidal_tmp_xy**2 + 2*tidal_tmp_xz**2 + 2*tidal_tmp_yz**2)**(1/2)
        tidal_zz_all /= (tidal_tmp_xx**2 + tidal_tmp_yy**2 + tidal_tmp_zz**2 + 2*tidal_tmp_xy**2 + 2*tidal_tmp_xz**2 + 2*tidal_tmp_yz**2)**(1/2)
    
    return tidal_xx_all, tidal_xy_all, tidal_xz_all, tidal_yy_all, tidal_yz_all
