#distutils: language = c++
#cython: language_level=3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 13:53:13 2021

@author: tibor
"""

import sys
from scipy.ndimage import labeled_comprehension
from scipy.linalg.cython_lapack cimport zheevr
cimport openmp
from cython cimport view
from scipy.fft import fftn, ifftn, fftfreq
import numpy as np
from libc.stdio cimport printf
from libcpp.map cimport map
import skimage.morphology
import config
from print_msg import print_status
cimport numpy as cnp # Importing parts of NumPY C-API
# Cython internally would handle this ambiguity so that the user would not need to use different names.
cimport cython 
cnp.import_array()  # So numpy's C API won't segfault
from functools import partial
from pathos.multiprocessing import ProcessingPool as Pool
include "array_defs.pxi"
rank = 0

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void ZHEEVR(complex[::1,:] H, double * eigvals,
                    complex[::1,:] Z, int nrows):
    """
    Computes the eigenvalues and vectors of a dense Hermitian matrix.
    Eigenvectors are returned in Z.
    Parameters
    ----------
    H : array_like
        Input Hermitian matrix.
    eigvals : array_like
        Input array to store eigenvalues.
    Z : array_like
        Output array of eigenvectors.
    nrows : int
        Number of rows in matrix.
    """
    cdef char jobz = b'V'
    cdef char rnge = b'A'
    cdef char uplo = b'L'
    cdef double vl=1, vu=1, abstol=0
    cdef int il=1, iu=1
    cdef int lwork = 18 * nrows
    cdef int lrwork = 24*nrows, liwork = 10*nrows
    cdef int info=0, M=0
    #These nee to be freed at the end
    cdef int * isuppz = <int *>PyDataMem_NEW((2*nrows) * sizeof(int))
    cdef complex * work = <complex *>PyDataMem_NEW(lwork * sizeof(complex))
    cdef double * rwork = <double *>PyDataMem_NEW((24*nrows) * sizeof(double))
    cdef int * iwork = <int *>PyDataMem_NEW((10*nrows) * sizeof(int))

    zheevr(&jobz, &rnge, &uplo, &nrows, &H[0,0], &nrows, &vl, &vu, &il, &iu, &abstol, &M, eigvals, &Z[0,0], &nrows, isuppz, work, &lwork, rwork, &lrwork, iwork, &liwork, &info)
    PyDataMem_FREE(work)
    PyDataMem_FREE(rwork)
    PyDataMem_FREE(isuppz)
    PyDataMem_FREE(iwork)
    if info != 0:
        if info < 0:
            raise Exception("Error in parameter : %s" & abs(info))
        else:
            raise Exception("Algorithm failed to converge")

def getFraction1Cut(float[:,:,:] overall_signature, float[:,:,:] rgrid, float[:] VIR_VEC, float delta_times_mean, int CONNECTIVITY, int N, int cut_idx):
    """
    Calculates the fraction of valid regions for a fixed virialization density
    If a region's mean density is above the density Delta, then the region is considered as valid.
    External Arguments:
    -------------
    overall_signature: (N, N, N)-shaped array
    This encompasses the smoothing-scale-maximized cluster signature.
    delta_times_mean: float
    This is the mean density of the universe times the virialization overdensity.
    Hence the mean density of a region needs to be above this value.
    cut_idx: integer
    CONNECTIVITY: integer, 
    Maximum number of orthogonal hops to consider a voxel as a neighbor
    The signature cut integer: All grid points with signatures below this value shall 
    be excluded from the discussion of "regions"
    Returns:
    ------------
    Float: The fraction of valid regions for this signature cut of index cut_idx
    """
    # The following grid points have signature values above the cut = qualify
    x,y,z = np.where(overall_signature.base > VIR_VEC[cut_idx])
    dense = np.zeros((N,N,N), dtype=bool)
    dense[x,y,z] = True
    labeled = skimage.morphology.label(dense, connectivity=CONNECTIVITY)
    nb_regions = labeled.max()
    if nb_regions == 0:
        return (1, cut_idx)
    else:
        # Calculate average density
        av_dens = labeled_comprehension(rgrid.base, labeled, np.arange(1,nb_regions+1), np.mean, np.float32, 0)
        # Calculate the fraction of valid regions
        frac = 0
        for region_idx in range(nb_regions):
            if av_dens[region_idx] >= delta_times_mean:
                frac += 1
        return (frac/nb_regions, cut_idx)
    
def getM(float[:,:,:] overall_signature, float[:,:,:] rgrid, float[:] SIG_VEC, float L_BOX, int N, int s_idx):
    """
    Calculates the mass of all the particles that have signature larger or equal to self.SIG_VEC[s_idx]
    Assumption: The volume of one particle is taken to be (self.L_BOX/self.N)**3
    External Arguments:
    -------------
    overall_signature: (N, N, N)-shaped array
    This is the smoothing-scale-maximized signature.
    s_idx: Integer
    The signature cut index: All grid points with signatures below self.SIG_VEC[s_idx] shall be excluded
    xsize, ysize, zsize: Integers
    Returns:
    -------------
    Float: The total mass of all the particles that have signature larger or equal to self.SIG_VEC[s_idx]
    """
    x, y, z = np.where(overall_signature.base > SIG_VEC[s_idx])
    rho = np.sum(rgrid.base[x,y,z])
    return (s_idx, rho*(L_BOX/N)**3)

cdef float[:,:,:,:,:] getHessianRealSpace(float complex [:,:,:] kloggrid, float [:,:,:] rgrid, float R): # Why not part of the NEXUS class? Serializing all of the object's self takes much time.
    """ Calculates the full (also last axis non-truncated) real-space Hessian
    Explicit Arguments:
    ------------
    kloggrid: (NxNxN) np.complex128, base-10 logarithm of momentum space density
    rgrid: (NxNxN) float array, grid density
    R: Float, smoothing scale
    Returns:
    ------------
    A (3, 3, N, N, N)-shaped array, encoding the real space Hessian of the smoothed grid, 
    with its first 2 indices (i and j) accessing the (i,j)-component of the Hessian
    """
    # Smooth logarithm of field in k-space
    cdef float[:,:,:] rsmoothgrid = np.zeros((config.N, config.N, config.N), dtype=np.float32)
    cdef float C_R = 0.0 # Ensures that the arithmetic mean of the input field remains unmodified
    cdef float[:] k = fftfreq(config.N, config.L_BOX/config.N).astype(np.float32) # Has negative Nyquist frequency
    kx,ky,kz = np.meshgrid(k.base,k.base,k.base) # Cannot be cythonized
    k_mesh = ky,kx,kz # Want first object ky to be independent of j and k argument: ky[i,j,k]...
    cdef float[:,:,:] k3d_squared = k_mesh[0]**2+k_mesh[1]**2+k_mesh[2]**2    
    rsmoothgrid = 10**(ifftn(kloggrid.base*np.exp(-k3d_squared.base*R**2*4*np.pi**2/2)).real.astype(np.float32))
    C_R = rgrid.base.sum()/rsmoothgrid.base.sum()
    rsmoothgrid = rsmoothgrid.base*C_R
    cdef float complex[:,:,:] ksmoothgrid = fftn(rsmoothgrid.base).astype(np.complex64)
                
    # Calculate Hessian of log-smoothed field
    cdef float complex[:,:,:] khess = np.zeros((config.N, config.N, config.N), dtype=np.complex64)
    cdef float[:,:,:,:,:] rhess = np.zeros((3, 3, config.N, config.N, config.N), dtype=np.float32)   
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    for i in range(3):
        for j in range(3):
            if i <= j:
                khess[:,:,:] = 0.0
                khess = -4*np.pi**2*ksmoothgrid.base*R**2*k_mesh[i]*k_mesh[j]
                rhess.base[i,j,:,:,:] = ifftn(khess.base).real.astype(np.float32)
                if i != j: # The Hessian is symmetric
                    rhess.base[j,i,:,:,:] = rhess.base[i,j,:,:,:]  
    # Clean-up
    del khess
    del rsmoothgrid
    del ksmoothgrid
    del kx; del ky; del kz; del k_mesh
    
    return rhess.base

@cython.boundscheck(False)
@cython.wraparound(False)
cdef float cython_abs(float x) nogil:
    if x >= 0.0:
        return x
    if x < 0.0:
        return -x

@cython.boundscheck(False)
@cython.wraparound(False)    
cdef float cython_heaviside(float x) nogil:
    if x >= 0.0:
        return 1
    else:
        return 0

@cython.boundscheck(False)
@cython.wraparound(False)
cdef float getClusterSig(int x, int y, int z, complex[::1,:] rhess, double[::1] eigval_tmp, complex[::1,:] eigvec_tmp, start_time):
    """
    Calculates the cluster signature at point (x,y,z)
    Explicit Arguments:
    ------------
    rhess: A (3, 3)-shaped array, encoding the real space Hessian of the smoothed grid
    eigval_tmp: A (3,)-shaped array, will be populated by the 3 eigenvalues
    eigval_vec: A (3,3)-shaped array, will be populated by the 3 eigenvectors
    Returns:
    ------------
    Cluster sig at point (x,y,z)
    """
    cdef float eig0
    cdef float eig1
    cdef float eig2
    eigval_tmp[:] = 0.0
    eigvec_tmp[:,:] = 0.0
    
    try:
        ZHEEVR(rhess[:,:], &eigval_tmp[0], eigvec_tmp, 3)
        # The eigenvalues are returned in ascending order, but not repeated according to their multiplicity.
        # However, it is essentially impossible to get the same eigenvalues twice.
    except (Exception, TypeError) as e:
        print_status(rank, start_time, "Eigenvalue problem failed at point {0}, {1}, {2}. Reason: {3}".format(x, y, z, e), allowed_any = True)
    
    eig0 = <float>(eigval_tmp[0]); eig1 = <float>(eigval_tmp[1]); eig2 = <float>(eigval_tmp[2])
    if eig0 == 0.0: # To avoid zero divisions in the signature calculations
        eig0 = 0.000001
    if eig1 == 0.0:
        eig1 = 0.000001
    if eig2 == 0.0:
        eig2 = 0.000001
    return cython_abs(eig2/eig0)*cython_abs(eig2)*cython_heaviside(-eig0)*cython_heaviside(-eig1)*cython_heaviside(-eig2)
     
@cython.boundscheck(False)
@cython.wraparound(False)
cdef float getFilSig(int x, int y, int z, complex[::1,:] rhess, double[::1] eigval_tmp, complex[::1,:] eigvec_tmp, start_time):
    """
    Calculates the filament signature at point (x,y,z)
    Explicit Arguments:
    ------------
    rhess: A (3, 3)-shaped array, encoding the real space Hessian of the smoothed grid
    eigval_tmp: A (3,)-shaped array, will be populated by the 3 eigenvalues
    eigval_vec: A (3,3)-shaped array, will be populated by the 3 eigenvectors
    Returns:
    ------------
    Fil sig at point (x,y,z)
    """
    cdef float eig0
    cdef float eig1
    cdef float eig2
    eigval_tmp[:] = 0.0
    eigvec_tmp[:,:] = 0.0
    
    try:
        ZHEEVR(rhess[:,:], &eigval_tmp[0], eigvec_tmp, 3)
        # The eigenvalues are returned in ascending order, but not repeated according to their multiplicity.
        # However, it is essentially impossible to get the same eigenvalues twice.
    except (Exception, TypeError) as e:
        print_status(rank, start_time, "Eigenvalue problem failed at point {0}, {1}, {2}. Reason: {3}".format(x, y, z, e), allowed_any = True)
    
    eig0 = <float>(eigval_tmp[0]); eig1 = <float>(eigval_tmp[1]); eig2 = <float>(eigval_tmp[2])
    if eig0 == 0.0: # To avoid zero divisions in the signature calculations
        eig0 = 0.000001
    if eig1 == 0.0:
        eig1 = 0.000001
    if eig2 == 0.0:
        eig2 = 0.000001
    return cython_abs(eig1/eig0)*(1-cython_abs(eig2/eig0))*cython_heaviside(1-cython_abs(eig2/eig0))*cython_abs(eig1)*cython_heaviside(-eig0)*cython_heaviside(-eig1)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef float getWallSig(int x, int y, int z, complex[::1,:] rhess, double[::1] eigval_tmp, complex[::1,:] eigvec_tmp, start_time):
    """
    Calculates the wall signature at point (x,y,z)
    Explicit Arguments:
    ------------
    rhess: A (3, 3)-shaped array, encoding the real space Hessian of the smoothed grid
    eigval_tmp: A (3,)-shaped array, will be populated by the 3 eigenvalues
    eigval_vec: A (3,3)-shaped array, will be populated by the 3 eigenvectors
    Returns:
    ------------
    Wall sig at point (x,y,z)
    """
    cdef float eig0
    cdef float eig1
    cdef float eig2
    eigval_tmp[:] = 0.0
    eigvec_tmp[:,:] = 0.0
    
    try:
        ZHEEVR(rhess[:,:], &eigval_tmp[0], eigvec_tmp, 3)
        # The eigenvalues are returned in ascending order, but not repeated according to their multiplicity.
        # However, it is essentially impossible to get the same eigenvalues twice.
    except (Exception, TypeError) as e:
        print_status(rank, start_time, "Eigenvalue problem failed at point {0}, {1}, {2}. Reason: {3}".format(x, y, z, e), allowed_any = True)
    
    eig0 = <float>(eigval_tmp[0]); eig1 = <float>(eigval_tmp[1]); eig2 = <float>(eigval_tmp[2])
    if eig0 == 0.0: # To avoid zero divisions in the signature calculations
        eig0 = 0.000001
    if eig1 == 0.0:
        eig1 = 0.000001
    if eig2 == 0.0:
        eig2 = 0.000001
    return (1-cython_abs(eig1/eig0))*cython_heaviside(1-cython_abs(eig1/eig0))*(1-cython_abs(eig2/eig0))*cython_heaviside(1-cython_abs(eig2/eig0))*cython_abs(eig0)*cython_heaviside(-eig0)

@cython.boundscheck(False)
@cython.wraparound(False)   
def getClusterSignatureOneScale(float complex [:,:,:] kloggrid, float [:,:,:] rgrid, start_time, int Ridx):
    """
    Calculates the cluster signature at each grid point for a given scale config.RSMOOTH[Ridx]
    Explicit arguments:
    ------------
    Ridx: integer, represents smoothing scale index
    Returns:
    ------------
    A ((N, N, N), Ridx)-tuple, first entry encodes the cluster signature
    """
    print_status(rank, start_time, "Working on smoothing scale {0}".format(config.RSMOOTH[Ridx]), allowed_any = True)
    cdef float[::1,:,:,:,:] rhess = np.asfortranarray(getHessianRealSpace(kloggrid, rgrid, config.RSMOOTH[Ridx]).base)
    print_status(rank, start_time, "Gotten Hessian of grav. potential", allowed_any = True)
    cdef float[:,:,:] cluster_sig = np.ones((config.N, config.N, config.N), dtype=np.float32)
    cdef double[::1,:] eigval_tmp = np.zeros((3,openmp.omp_get_max_threads()), dtype=np.float64, order='F')
    cdef complex[::1,:,:] eigvec_tmp = np.zeros((3,3,openmp.omp_get_max_threads()), dtype=np.complex128, order='F')
    cdef int N = config.N
    cdef Py_ssize_t x_idx
    cdef Py_ssize_t y_idx
    cdef Py_ssize_t z_idx
    for x_idx in range(N):
        for y_idx in range(N):
            for z_idx in range(N):
                cluster_sig[x_idx, y_idx, z_idx] = getClusterSig(x_idx, y_idx, z_idx, rhess.base[:,:,x_idx, y_idx, z_idx].astype(np.complex128), eigval_tmp[:,openmp.omp_get_thread_num()], eigvec_tmp[:,:,openmp.omp_get_thread_num()], start_time)
    print_status(rank, start_time, "Gotten cluster signatures at every grid point", allowed_any = True)
    
    # Clean-up
    del rhess; del eigval_tmp; del eigvec_tmp
    return (cluster_sig.base, Ridx)

@cython.boundscheck(False)
@cython.wraparound(False)   
def getFilamentSignatureOneScale(float complex [:,:,:] kloggrid, float [:,:,:] rgrid, start_time, int Ridx):
    """
    Calculates the cluster signature at each grid point for a given scale config.RSMOOTH[Ridx]
    Explicit arguments:
    ------------
    Ridx: integer, represents smoothing scale index
    Returns:
    ------------
    A ((N, N, N), Ridx)-tuple, first entry encodes the filament signature
    """
    print_status(rank, start_time, "Working on smoothing scale {0}".format(config.RSMOOTH[Ridx]), allowed_any = True)
    cdef float[::1,:,:,:,:] rhess = np.asfortranarray(getHessianRealSpace(kloggrid, rgrid, config.RSMOOTH[Ridx]).base)
    print_status(rank, start_time, "Gotten Hessian of grav. potential", allowed_any = True)
    cdef float[:,:,:] fil_sig = np.ones((config.N, config.N, config.N), dtype=np.float32)
    cdef double[::1,:] eigval_tmp = np.zeros((3,openmp.omp_get_max_threads()), dtype=np.float64, order='F')
    cdef complex[::1,:,:] eigvec_tmp = np.zeros((3,3,openmp.omp_get_max_threads()), dtype=np.complex128, order='F')
    cdef int N = config.N
    cdef Py_ssize_t x_idx
    cdef Py_ssize_t y_idx
    cdef Py_ssize_t z_idx
    for x_idx in range(N):
        for y_idx in range(N):
            for z_idx in range(N):
                fil_sig[x_idx, y_idx, z_idx] = getFilSig(x_idx, y_idx, z_idx, rhess.base[:,:,x_idx, y_idx, z_idx].astype(np.complex128), eigval_tmp[:,openmp.omp_get_thread_num()], eigvec_tmp[:,:,openmp.omp_get_thread_num()], start_time)
    print_status(rank, start_time, "Gotten fil signatures at every grid point", allowed_any = True)
    
    # Clean-up
    del rhess; del eigval_tmp; del eigvec_tmp
    return (fil_sig.base, Ridx)

@cython.boundscheck(False)
@cython.wraparound(False)   
def getWallSignatureOneScale(float complex [:,:,:] kloggrid, float [:,:,:] rgrid, start_time, int Ridx):
    """
    Calculates the cluster signature at each grid point for a given scale config.RSMOOTH[Ridx]
    Explicit arguments:
    ------------
    Ridx: integer, represents smoothing scale index
    Returns:
    ------------
    A ((N, N, N), Ridx)-tuple, first entry encodes the cluster signature
    """
    print_status(rank, start_time, "Working on smoothing scale {0}".format(config.RSMOOTH[Ridx]), allowed_any = True)
    cdef float[::1,:,:,:,:] rhess = np.asfortranarray(getHessianRealSpace(kloggrid, rgrid, config.RSMOOTH[Ridx]).base)
    print_status(rank, start_time, "Gotten Hessian of grav. potential", allowed_any = True)
    cdef float[:,:,:] wall_sig = np.ones((config.N, config.N, config.N), dtype=np.float32)
    cdef double[::1,:] eigval_tmp = np.zeros((3,openmp.omp_get_max_threads()), dtype=np.float64, order='F')
    cdef complex[::1,:,:] eigvec_tmp = np.zeros((3,3,openmp.omp_get_max_threads()), dtype=np.complex128, order='F')
    cdef int N = config.N
    cdef Py_ssize_t x_idx
    cdef Py_ssize_t y_idx
    cdef Py_ssize_t z_idx
    for x_idx in range(N):
        for y_idx in range(N):
            for z_idx in range(N):
                wall_sig[x_idx, y_idx, z_idx] = getWallSig(x_idx, y_idx, z_idx, rhess.base[:,:,x_idx, y_idx, z_idx].astype(np.complex128), eigval_tmp[:,openmp.omp_get_thread_num()], eigvec_tmp[:,:,openmp.omp_get_thread_num()], start_time)
    print_status(rank, start_time, "Gotten wall signatures at every grid point", allowed_any = True)
    
    # Clean-up
    del rhess; del eigval_tmp; del eigvec_tmp
    return (wall_sig.base, Ridx)




cdef class NEXUS: 
    
    cdef int N
    cdef float L_BOX
    cdef float[:] RSMOOTH
    cdef float NORM  
    cdef float DELTA
    cdef float[:] SIG_VEC
    cdef float[:] VIR_VEC
    cdef float[:,:,:] rgrid
    
    # Setting instance variables
    def __init__(self, int N, float L_BOX, float[:] RSMOOTH, float NORM, float[:] SIG_VEC, float[:] VIR_VEC, float DELTA):
        # Grid resolution
        self.N = N
        # Extension in x-, y- and z-direction in comoving Mpc/h, float
        self.L_BOX = L_BOX
        # Smoothing scales, float
        self.RSMOOTH = RSMOOTH
        # This is the arithmetic mean of the 3 scale-independent signature grids
        self.NORM = NORM
        # Signatures S_f, S_w of interest for mass threshold method. Logarithmically spaced, but not logarithm thereof
        self.SIG_VEC = SIG_VEC
        # Signatures S_c of interest for vir threshold method. Logarithmically spaced, but not logarithm thereof
        self.VIR_VEC = VIR_VEC
        # DELTA: virial density for virialization threshold method
        self.DELTA = DELTA
        
    # Set and get functions
    def setRgrid(self, rgrid):
        self.rgrid = rgrid
    
    def getRgrid(self):
        return self.rgrid
    
    def getSIG_VEC(self):
        return self.SIG_VEC
    
    def getVIR_VEC(self):
        return self.VIR_VEC

    def getClusterSignatureVariousScales(self, start_time):
        """
        Calculates the cluster signatures for a range of smoothing scales
        Returns:
        -------------
        (n_max, N, N, N)-shaped array, encoding the cluster signature at various smoothing scales
        """
        # Calculate FFT of grid and add instance variable kloggrid
        self.rgrid.base[self.rgrid.base==float(0)] = 1e-5 # Checking for zero-entries due to logarithm
        cdef float complex[:,:,:] kloggrid = fftn(np.log10(self.rgrid.base)).astype(np.complex64)
                
        # Prepare pooled job submission
        cdef float[:,:,:,:] cluster_sig = np.zeros((self.RSMOOTH.shape[0], self.N, self.N, self.N), dtype=np.float32)
        
        # Find cluster signature
        with Pool(processes=openmp.omp_get_max_threads()) as pool:
            results = pool.map(partial(getClusterSignatureOneScale, kloggrid.base, self.rgrid.base, start_time), [i for i in range(0, self.RSMOOTH.shape[0])])
        for result in results:
            cluster_sig_new, R_idx = tuple(result)
            cluster_sig.base[R_idx] = cluster_sig_new
        cdef float normalize = self.NORM*(self.N**3/cluster_sig.base.sum())  
        
        # Clean-up
        del kloggrid
        
        return cluster_sig.base*normalize
    
    def getFilamentSignatureVariousScales(self, start_time):
        """
        Calculates the filament signatures for a range of smoothing scales
        Returns:
        -------------
        (self.RSMOOTH.shape[0], N, N, N)-shaped array, encoding the filament signature at various smoothing scales
        """
        # Calculate FFT of grid and update instance variable kloggrid
        self.rgrid.base[self.rgrid.base==float(0)] = 1e-5 # Checking for zero-entries due to logarithm
        cdef float complex[:,:,:] kloggrid = fftn(np.log10(self.rgrid.base)).astype(np.complex64)
                
        # Prepare pooled job submission
        cdef float[:,:,:,:] fil_sig = np.zeros((self.RSMOOTH.shape[0], self.N, self.N, self.N), dtype=np.float32)
        
        # Pooled job submission for cluster signature
        with Pool(processes=openmp.omp_get_max_threads()) as pool:
            results = pool.map(partial(getFilamentSignatureOneScale, kloggrid.base, self.rgrid.base, start_time), [i for i in range(0, self.RSMOOTH.shape[0])])
        for result in results:
            fil_sig_new, R_idx = tuple(result)
            fil_sig.base[R_idx] = fil_sig_new
        cdef float normalize = self.NORM*(self.N**3/fil_sig.base.sum())
        
        # Clean-up
        del kloggrid
        
        return fil_sig.base*normalize
    
    def getWallSignatureVariousScales(self, start_time):
        """
        Calculates the wall signatures for a range of smoothing scales
        Returns:
        -------------
        (self.RSMOOTH.shape[0], N, N, N)-shaped array, encoding the wall signature at various smoothing scales
        """
        # Calculate FFT of grid and update instance variable kloggrid
        self.rgrid.base[self.rgrid.base==float(0)] = 1e-5 # Checking for zero-entries due to logarithm
        cdef float complex[:,:,:] kloggrid = fftn(np.log10(self.rgrid.base)).astype(np.complex64)
                
        # Prepare pooled job submission
        cdef float[:,:,:,:] wall_sig = np.zeros((self.RSMOOTH.shape[0], self.N, self.N, self.N), dtype=np.float32)
        
        # Pooled job submission for wall signature
        with Pool(processes=openmp.omp_get_max_threads()) as pool:
            results = pool.map(partial(getWallSignatureOneScale, kloggrid.base, self.rgrid.base, start_time), [i for i in range(0, self.RSMOOTH.shape[0])])
        for result in results:
            wall_sig_new, R_idx = tuple(result)
            wall_sig.base[R_idx] = wall_sig_new
        cdef float normalize = self.NORM*(self.N**3/wall_sig.base.sum())
        
        # Clean-up
        del kloggrid
        
        return wall_sig.base*normalize
    
    def getDeltaMSquared(self, float[:,:,:] overall_signature):
        """
        Calculates the quantity Delta M Squared, from NEXUS paper
        External Arguments:
        -------------
        overall_signature: (N, N, N)-shaped array
        This is the smoothing-scale-maximized signature.
        Returns:
        -------------
        Float 1D array: The quantity Delta M Squared evaluated at discrete points in the range self.SIG_VEC
        """
        cdef float[:] m_vec = np.zeros(len(self.SIG_VEC), dtype = np.float32)
        cdef float[:] delta_m_squared = np.zeros(len(self.SIG_VEC), dtype = np.float32)
        with Pool(processes=openmp.omp_get_max_threads()) as pool:
            results = pool.map(partial(getM, overall_signature.base, self.rgrid.base, self.SIG_VEC.base, np.float32(config.L_BOX), config.N), [i for i in range(len(self.SIG_VEC))])
        for result in results:
            s_idx, M = tuple(result)
            m_vec[s_idx] = M                
        for s_idx in range(len(self.SIG_VEC)):
            if s_idx == 0:
                delta_m_squared[s_idx] = np.abs((m_vec[s_idx+1]**2-m_vec[s_idx]**2)/(np.log(self.SIG_VEC[s_idx+1])-np.log(self.SIG_VEC[s_idx])))
            elif s_idx == len(self.SIG_VEC) - 1:
                delta_m_squared[s_idx] = np.abs(-(m_vec[s_idx-1]**2-m_vec[s_idx]**2)/(np.log(self.SIG_VEC[s_idx])-np.log(self.SIG_VEC[s_idx-1])))
            else:
                delta_m_squared[s_idx] = 1/2*(np.abs((m_vec[s_idx+1]**2-m_vec[s_idx]**2)/(np.log(self.SIG_VEC[s_idx+1])-np.log(self.SIG_VEC[s_idx]))) + np.abs(-(m_vec[s_idx-1]**2-m_vec[s_idx]**2)/(np.log(self.SIG_VEC[s_idx])-np.log(self.SIG_VEC[s_idx-1]))))
        return delta_m_squared.base

    def getFractionOfValidRegions(self, float[:,:,:] overall_signature):
        """
        Calculates the fraction of valid regions according to the NEXUS paper algorithm:
        If a region's mean density is above the density self.DELTA, then the region is considered as valid.
        External Arguments:
        -------------
        overall_signature: (N, N, N)-shaped array,
        This encompasses the smoothing-scale-maximized signature.
        tol: integer
        The tolerance in defining adjacency
        Returns: 
        -------------
        Float: The fraction of valid regions as an array
        """
        assert sum(self.VIR_VEC[cut_idx] < 0 for cut_idx in range(len(self.VIR_VEC))) == 0, \
            "Error! Please choose positive semidefinite signature cuts only, to avoid meaningless calculations!"
        cdef float[:] frac_vec = np.zeros(len(self.VIR_VEC), dtype = np.float32)
        cdef float delta_times_mean = self.DELTA*self.rgrid.base.sum()/self.N**3
        cdef Py_ssize_t sc_idx
        with Pool(processes=openmp.omp_get_max_threads()) as pool:
            results = pool.map(partial(getFraction1Cut, overall_signature.base, self.rgrid.base, self.VIR_VEC.base, delta_times_mean, config.CONNECTIVITY, config.N), [i for i in range(len(self.VIR_VEC))])
        for result in results:
            frac_new, sc_idx = tuple(result)
            frac_vec.base[sc_idx] = frac_new
        return frac_vec.base