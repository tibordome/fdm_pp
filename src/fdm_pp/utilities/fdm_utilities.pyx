#distutils: language = c++
#cython: language_level=3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 11:10:47 2021

@author: tibor
"""

# from https://stackoverflow.com/questions/55453110/how-to-find-local-maxima-of-3d-array-in-python

import numpy as np
cimport numpy as cnp
from scipy import ndimage as ndi
from libc.math cimport sqrt
from scipy import interpolate
from cython.parallel import parallel, prange, threadid
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
import config
from numpy cimport PyArray_ZEROS  
cnp.import_array()  # So numpy's C API won't segfault

def getPotential( rho, rhobar, Lbox, G ):
	
    N = rho.shape[0]

    # fourier space variables
    klin = (2*np.pi/Lbox) * np.arange(-N/2,N/2)
    kx, ky, kz = np.meshgrid(klin, klin, klin, indexing='ij')
    kSq = kx**2 + ky**2 + kz**2
    kSq = np.fft.fftshift(kSq) # this brings it into the fft output format [0, pos, neg] frequ
    kSq[kSq==0]=1 # There is no infinite wavelength mode in a finite box...

    # Poisson solver
    V = -(np.fft.fftn( 4*np.pi*G * (rho-rhobar) )) / kSq  # (Vhat)
    V = np.fft.ifftn(V)

    # normalize so mean potential is 0
    V -= np.mean(V) # Hence the infinite wavelength mode is irrelevant
	
    V = np.real(V)
	#print(np.min(V))
	#print(np.max(V))
	
    return V


def getLocalMaxima3D(data, order=1):
    """Detects local maxima in a 3D array

    Parameters
    ---------
    data : 3d ndarray
    order : int
        How many points on each side to use for the comparison

    Returns
    -------
    coords : ndarray
        coordinates of the local maxima
    values : ndarray
        values of the local maxima
    """
    size = 1 + 2 * order
    footprint = np.ones((size, size, size))
    footprint[order, order, order] = 0

    filtered = ndi.maximum_filter(data, footprint=footprint, mode="wrap") # Max filter: you take max, not average!. Note that 'wrap' amounts to PBC.
    mask_local_maxima = data > filtered

    coords = np.asarray(np.where(mask_local_maxima)).T
    values = data[mask_local_maxima]

    return coords, values

    
def getEnclosedMassProfiles(int[:,:] coords, float[:,:,:] rho, int Npeaks):
    """Calculates spherical enclosed mass profiles for each local minimum (peak below) in the
    gravitational potential V
    Arguments:
    --------------
    coords: (Npeaks x 3) ints, peak indices
    rho: (N x N x N) floats, density field
    Npeaks: int, number of peaks
    Returns:
    --------------
    rho_enc, R_enc, M_enc: each (Npeaks x Nchar) floats, enclosed density, radius and mass profiles
    These are all calculated out to Nchar * config.DEL_X distance"""
    
    # Preparatory definitions
    R_enc = np.tile(np.reshape(np.arange(1,config.NCHAR+1)*config.DEL_X, (1,config.NCHAR)), (1,1)) # Real-space, one row vector
    cdef int perrank = Npeaks//size
    cdef int nb_peaks = perrank
    cdef int rank_ = rank
    cdef bint last = rank_ == size - 1 # Whether or not last process
    if last:
        nb_peaks += last*(Npeaks-(rank_+1)*perrank)
    comm.Barrier()
    cdef float[:,:] M_enc = np.zeros((nb_peaks, config.NCHAR), dtype=np.float32)
    cdef int[:,:,:,:] iipy = np.zeros((nb_peaks, 2*config.NCHAR+1, 2*config.NCHAR+1, 2*config.NCHAR+1), dtype=np.int32)
    cdef int[:,:,:,:] jjpy = np.zeros((nb_peaks, 2*config.NCHAR+1, 2*config.NCHAR+1, 2*config.NCHAR+1), dtype=np.int32)
    cdef int[:,:,:,:] kkpy = np.zeros((nb_peaks, 2*config.NCHAR+1, 2*config.NCHAR+1, 2*config.NCHAR+1), dtype=np.int32)
    cdef int[:,:,:,:] ii = np.zeros((nb_peaks, 2*config.NCHAR+1, 2*config.NCHAR+1, 2*config.NCHAR+1), dtype=np.int32)
    cdef int[:,:,:,:] jj = np.zeros((nb_peaks, 2*config.NCHAR+1, 2*config.NCHAR+1, 2*config.NCHAR+1), dtype=np.int32)
    cdef int[:,:,:,:] kk = np.zeros((nb_peaks, 2*config.NCHAR+1, 2*config.NCHAR+1, 2*config.NCHAR+1), dtype=np.int32)
    cdef int p_idx
    cdef int N = config.N
    cdef int NCHAR = config.NCHAR
    cdef float DEL_X = config.DEL_X
    
    # Precalculate meshgrid entries
    for p in range(rank_*perrank, (rank_+1)*perrank+last*(Npeaks-(rank_+1)*perrank)):
        [iipy.base[p-rank_*perrank], jjpy.base[p-rank_*perrank], kkpy.base[p-rank_*perrank]] = np.meshgrid(np.arange(coords[p, 0]-config.NCHAR,coords[p, 0]+config.NCHAR+1), np.arange(coords[p, 1]-config.NCHAR,coords[p, 1]+config.NCHAR+1), np.arange(coords[p, 2]-config.NCHAR,coords[p, 2]+config.NCHAR+1), indexing='ij') 
        ii.base[p-rank_*perrank] = iipy.base[p-rank_*perrank] % config.N # Shape is (2*config.NCHAR+1, 2*config.NCHAR+1, 2*config.NCHAR+1)
        jj.base[p-rank_*perrank] = jjpy.base[p-rank_*perrank] % config.N
        kk.base[p-rank_*perrank] = kkpy.base[p-rank_*perrank] % config.N
    
    # Multithreading & MPI
    for p_idx in prange(rank_*perrank, (rank_+1)*perrank+last*(Npeaks-(rank_+1)*perrank), schedule = "dynamic", nogil=True):
        M_enc[p_idx-rank_*perrank] = getEnclosedMassProfile(coords, rho, NCHAR, N, DEL_X, M_enc[p_idx-rank_*perrank], iipy[p_idx-rank_*perrank], jjpy[p_idx-rank_*perrank], kkpy[p_idx-rank_*perrank], ii[p_idx-rank_*perrank], jj[p_idx-rank_*perrank], kk[p_idx-rank_*perrank], p_idx)
    rho_enc = np.zeros((nb_peaks, config.NCHAR), dtype=np.float32)
    for p in range(nb_peaks):
        rho_enc[p] = np.divide(M_enc[p], 4/3*np.pi*np.power(R_enc, 3))
    
    # Gather results from different ranks
    M_enc = np.reshape(M_enc.base, (1, nb_peaks*config.NCHAR))
    rho_enc = np.reshape(rho_enc, (1, nb_peaks*config.NCHAR))
    count_new = comm.gather(nb_peaks, root=0)
    count_new = comm.bcast(count_new, root = 0)
    recvcounts = np.array(count_new)*config.NCHAR
    rdispls = np.zeros_like(recvcounts)
    for j in range(rdispls.shape[0]):
        rdispls[j] = np.sum(recvcounts[:j])
    
    # Publish result on rank == 0
    M_enc_total = np.empty(Npeaks*config.NCHAR, dtype = np.float32)
    rho_enc_total = np.empty(Npeaks*config.NCHAR, dtype = np.float32)
    comm.Gatherv(M_enc, [M_enc_total, recvcounts, rdispls, MPI.FLOAT], root = 0)
    comm.Gatherv(rho_enc, [rho_enc_total, recvcounts, rdispls, MPI.FLOAT], root = 0)
    if rank == 0:
        M_enc_total = np.reshape(M_enc_total, (Npeaks, config.NCHAR))
        rho_enc_total = np.reshape(rho_enc_total, (Npeaks, config.NCHAR))
        R_enc = np.tile(np.reshape(np.arange(1,config.NCHAR+1)*config.DEL_X, (1,config.NCHAR)), (Npeaks,1))
    else:
        R_enc = np.empty(Npeaks*config.NCHAR, dtype = np.float32)
    return rho_enc_total, R_enc, M_enc_total # Only rank == 0 return is relevant


cdef float[:] getEnclosedMassProfile(int[:,:] coords, float[:,:,:] rho, int Nchar, int N, float DEL_X, float[:] M_enc, int[:,:,:] iipy, int[:,:,:] jjpy, int[:,:,:] kkpy, int[:,:,:] ii, int[:,:,:] jj, int[:,:,:] kk, int p_idx) nogil:
    cdef int r
    cdef int row1
    cdef int row2
    cdef int col
    cdef int xmin
    cdef int ymin
    cdef int zmin
    for r in range(1, Nchar+1):
        for row1 in range(2*r+1):
            for row2 in range(2*r+1):
                for col in range(2*r+1):
                    if sqrt((iipy[row1+Nchar-r, row2+Nchar-r, col+Nchar-r] - coords[p_idx, 0])**2 + (jjpy[row1+Nchar-r, row2+Nchar-r, col+Nchar-r] - coords[p_idx, 1])**2 + (kkpy[row1+Nchar-r, row2+Nchar-r, col+Nchar-r] - coords[p_idx, 2])**2) < r:
                        M_enc[r-1] += rho[ii[row1+Nchar-r, row2+Nchar-r, col+Nchar-r], jj[row1+Nchar-r, row2+Nchar-r, col+Nchar-r], kk[row1+Nchar-r, row2+Nchar-r, col+Nchar-r]]*DEL_X**3
    return M_enc

def myBinMean1D(X, W, y):
    """Weighted 1D histogram    
    Inputs: X - Nx1 data
            W - Nx1 weights
            y - vector of N2 bin centers along 1st dim of X
    Output: H - N2x1 histo, value for each bin center y"""
    xNumBins = y.shape[0]
    f = interpolate.interp1d(y, np.arange(0, xNumBins), kind = 'linear', bounds_error = False, fill_value="extrapolate")
    Xi = np.array([int(i) for i in f(X)]) # Mapping X to bin centers
    Xi = np.maximum(np.minimum(Xi, xNumBins-1),0) # Limit indices to the range [0, xNumBins-1]
    H = np.bincount(Xi, weights = W) # Count number of elements in each bin, with weight W
    norm = np.bincount(Xi) # Count (unweighted) number of elements in each bin
    assert H.shape[0] == xNumBins and norm.shape[0] == xNumBins
    H = np.divide(H, norm)
    return H

def getDensityProfiles(coords, rho, Npeaks, invalids):
    """Calculates density profiles out to config.L_BOX/2 distance
    For each peak, get config.N values with radial jump being config.DEL_X/2"""
    rBins = config.DEL_X/2*np.arange(1,config.N+1) # With this, we cover the entire grid (PBC)
    rho_profile = np.zeros((Npeaks, config.N))
    [ix, iy, iz] = np.meshgrid(np.arange(0,config.N), np.arange(0,config.N), np.arange(0,config.N), indexing='ij')
    for p in range(Npeaks):
        if p not in invalids:
            d = config.DEL_X *(np.reshape(ix, (config.N**3,))-coords[p,0]) # Has shape (config.N**3,)
            d[d<-config.L_BOX/2] = d[d<-config.L_BOX/2] + config.L_BOX
            d[d>config.L_BOX/2]  = d[d>config.L_BOX/2]  - config.L_BOX
            r2 = np.power(d,2)
            d = config.DEL_X *(np.reshape(iy, (config.N**3,))-coords[p,1])
            d[d<-config.L_BOX/2] = d[d<-config.L_BOX/2] + config.L_BOX
            d[d>config.L_BOX/2]  = d[d>config.L_BOX/2]  - config.L_BOX
            r2 = r2 + np.power(d,2)
            d = config.DEL_X *(np.reshape(iz, (config.N**3,))-coords[p,2])
            d[d<-config.L_BOX/2] = d[d<-config.L_BOX/2] + config.L_BOX
            d[d>config.L_BOX/2]  = d[d>config.L_BOX/2]  - config.L_BOX
            r2 = r2 + np.power(d,2)
            r = np.sqrt(r2) # Has shape (config.N**3,)
            
            H = myBinMean1D(r, np.reshape(rho, (config.N**3,)), rBins) # Shape is (config.N,)
        
            rho_profile[p,:] = H # Shape is (config.N,)
    return rho_profile
        
def getMDelta(rho_enc, rhobar, M_enc, R_enc, Npeaks):
    """Calculate M_Delta mass and R_Delta radius for each peak"""
    M_Delta = np.zeros((Npeaks,), dtype = np.float32)
    R_Delta = np.zeros((Npeaks,), dtype = np.float32)
    invalids = []
    for p in range(Npeaks):
        if rho_enc[p,0] > config.OVERDENSITY*rhobar:
            im = 0
            while rho_enc[p,im] > config.OVERDENSITY*rhobar:
                im = im +1
            r_intersect = (config.OVERDENSITY*rhobar - rho_enc[p,im-1]) / (rho_enc[p,im] - rho_enc[p,im-1]) * (R_enc[p,im] - R_enc[p,im-1])  +  R_enc[p,im-1]
            M_Delta[p] = (M_enc[p,im] - M_enc[p,im-1]) / (R_enc[p,im] - R_enc[p,im-1]) * (r_intersect - R_enc[p,im-1])  +  M_enc[p,im-1]
            R_Delta[p] = r_intersect
            assert r_intersect >= R_enc[p,im-1]
            assert r_intersect <= R_enc[p,im]
        else:
            invalids.append(p) # If smallest nbh of peak isn't all above config.OVERDENSITY*rhobar, invalid
    return M_Delta, R_Delta, invalids

    
def constructCat(float[:,:] dm_xyz, float[:,:] coms, float[:] R_Delta, int Npeaks):
    cdef int perrank = Npeaks//size
    cdef bint last = rank == size - 1 # Whether or not last process
    cdef int p_idx
    cdef int nb_peaks = perrank
    cdef int rank_ = rank
    if last:
        nb_peaks += last*(Npeaks-(rank_+1)*perrank)
    cdef int[:,:] shs_rank = np.zeros((nb_peaks, dm_xyz.shape[0]), dtype=np.int32)

    for p_idx in prange(rank_*perrank, (rank_+1)*perrank+last*(Npeaks-(rank_+1)*perrank), schedule = "dynamic", nogil=True):
        shs_rank[p_idx-rank_*perrank] = getEnclosedPtcs(dm_xyz, shs_rank[p_idx-rank_*perrank], coms[p_idx], R_Delta[p_idx])
        
    count_new = comm.gather(nb_peaks, root=0)
    count_new = np.array(comm.bcast(count_new, root = 0))
    sh_cat = [[] for i in range(nb_peaks)]
    for p_idx in range(nb_peaks):
        for dm_ptc in range(dm_xyz.shape[0]):
            if shs_rank[p_idx, dm_ptc] != 0:
                sh_cat[p_idx].append(int(shs_rank[p_idx, dm_ptc]-1))
            else:
                break
    sh_cat = comm.gather(sh_cat, root = 0)
    if rank == 0:
        sh_cat_full = [[] for i in range(Npeaks)]
        for r in range(size):
            for sh in range(count_new[r]):
                if r != 0:
                    add = np.array([count_new[i] for i in range(r)]).sum()
                else:
                    add = 0
                sh_cat_full[add+sh] += sh_cat[r][sh]
    else:
        sh_cat_full = None
    return sh_cat_full # Only rank = 0 content matters
    
cdef int[:] getEnclosedPtcs(float[:,:] dm_xyz, int[:] shs_rank, float[:] com, float R_Delta) nogil:
    cdef int count = 0
    cdef int dm_ptc
    for dm_ptc in range(dm_xyz.shape[0]):
        if (dm_xyz[dm_ptc,0]-com[0])**2+(dm_xyz[dm_ptc,1]-com[1])**2+(dm_xyz[dm_ptc,2]-com[2])**2 < R_Delta**2:        
            shs_rank[count] = dm_ptc+1
            count += 1
    return shs_rank