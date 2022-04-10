#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 11:53:13 2020

@author: tibor
"""
    
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
import h5py
from print_msg import print_status
import os
import numpy as np
import inspect

def initialize():   
    
    # Simulation Parameters
    global N # Resolution
    global L_BOX # Side of box in cMpc/h
    global DEL_X # N/L_BOX
    global DM_TYPE # string, always "fdm"
    global MASS_UNIT # AREPO mass unit, usually 1e+10 M_sun/h
    
    # Catalogue generation
    global FILE_TYPE # For generated viz files, note that only png and pdf are allowed
    global D_LOGSTART # Logarithm of smallest value for elliptical radius / RDelta
    global D_LOGEND # Logarithm of largest value for ellipsoidal radius / RDelta
    global D_BINS # Number of bins for elliptical radius discretization
    global NCHAR # The characteristic radius out to which the enclosed mass profile is calculated. <= int(N/2)-1, otherwise 2*NCHAR+1 would mean that you cover parts of box more than once
    global M_TOL # E1 convergence tolerance
    global N_WALL # Maximum number of iterations for convergence at given r_ell = d
    global N_MIN # Minimum number of particles (DM or star particle) in any iteration; if undercut, shape is unclassified
    global SAFE # Since E1 and S1 are sensitive to the outskirts of the object at hand (since shape tensor is not reduced), it 
    # is paramount not to leave out any outskirt particle in the outermost sphere/shell. This is achieved by adding some SAFE to max_i || COM - x_i|| to get d_max.
    global MIN_NUMBER_DM_PTCS # Minimum number of DM particles in CSH to qualify for catalogue entry
    global MIN_NUMBER_STAR_PTCS # Minimum number of star particles in galaxy to qualify for catalogue entry
    global SNAP_ABB # List of strings, 3 digits after snapdir_* to differentiate snapshots from one another
    global SNAP # Momentary snap that is being dealt with, always updated
    global CAT_DEST # Where to store catalogues of E1 results
    global VIZ_DEST # Where the visualizations of shapes, major axes are stored
    global HDF5_SNAP_DEST # Where we can find the snapshot
    global AXISYMMETRY_LIMIT # All A3D values above this value will lead to discarding of CSH/Gx
    global SNAP_MAX # Control the HDF5 reading of the output files
    global G # Gravitational constant for Poisson solver
    global OVERDENSITY
    global SAVE_FIGS # Whether or not to save figures (shape viz, discarded distros etc) to disk. 1 Fig. can be ~ 400 MB for N = 512 already!
    
    # Analysis
    global GRID_METHOD # How to calculate grid when overplotting major axes with DM distro
    global MEAN_BINS # Number of bins for mean +- stand_dev plots
    global MEAN_BINS_M # Number of bins for mean +- stand_dev plots when we split into mass bins
    global HALO_REGION # Where to calculate the DM CSH shape; either "Full" or "Inner"
    global M_BINS_C # Number of mass bins, if const_occ is chosen
    global M_SLICES # Number of mass bins will be M_SLICES + 1, if log_split is chosen
    global M_SPLIT_TYPE # Mass splitting type, either "const_occ" or "log_slice"
    global R_LOGSTART # Logarithm of smallest value for Euclidian distance when we split into radial bins for Q, S, T averaging
    global R_LOGEND # Logarithm of largest value for ellipsoidal radius / R200 when we split into radial bins for Q, S, T averaging
    global R_BIN # Number of radial bins for Q, S, T averaging
    global T_CUT_LOW # Float, Triaxiality cut, floor
    global T_CUT_HIGH # Float, Triaxiality cut, cap
    global R_SLICES # Number of radial bins will be R_SLICES + 1, if "proper_split" is chosen, instead of "just_r_smaller_2"
    global CUT_LEFT # For projecting major axes onto CIC density plot: z_min (z-dimension)
    global CUT_RIGHT # For projecting major axes onto CIC density plot: z_max (z-dimension)
    global ERROR_METHOD # How to calculate error bars on alignment measures, shape profiles and density profiles: "bootstrap" or "SEM" or "median_quantile"
    global HIST_NB_BINS # Number of histogram bins (for T, misalignment etc)
    global SHAPE_DEST # Where to store shape analysis results
    global ALIGNMENT_DEST # Where to store alignment analysis results
    global PROJECTION_DEST # Where to store projection images
    
    # Disperse
    global FAST_DISPERSE # Grid size in cMpc/h for efficient DISPERSE: -cut 0 run and -cut 5*RMS run.
    
    # NEXUS+
    global SIG_VEC # Logarithmic limits for signatures S_f, S_w
    global VIR_VEC # Logarithmic limits for signatures S_c
    global RSMOOTH # Smoothing scales, 1D array, Floats or Doubles
    global NORM # This is the arithmetic mean of the 3 scale-independent signature grids. We do not care about absolute signatures after all.
    global MASK # Whether or not to apply cluster mask for fil identification, cluster-fil mask for walls. NEXUS paper does not say.
    global CONNECTIVITY # Maximum number of orthogonal hops to consider a voxel as a neighbor. 3: diagonal connectivity allowed. 1: just orthogonal connectivity.
    global OMEGA_M # Fractional matter density today
    global OMEGA_L # Fractional dark energy density today
            
    # Simulation Parameters
    N = 1024
    L_BOX = float(1.7) # In cMpc/h, needs to be float32 for CIC for instance, so transform when the time comes
    DEL_X = L_BOX/N
    DM_TYPE = "fdm"
    MASS_UNIT = 1e+10
    
    # Alignment/Ellipticity Analysis Parameters
    GRID_METHOD = "CIC"
    MEAN_BINS = 7
    MEAN_BINS_M = 7
    HALO_REGION = "Inner"
    M_BINS_C = 2
    M_SLICES = 1
    M_SPLIT_TYPE = "log_slice"
    R_LOGSTART = -3
    R_LOGEND = 1
    R_BIN = 10
    T_CUT_LOW = 0.0
    T_CUT_HIGH = 1.0
    R_SLICES = 0
    ERROR_METHOD = "SEM" # Note that "bootstrap" makes little sense for SS and SP, since it is not 1 distro, 1 histogram.
    HIST_NB_BINS = 11
    
    # Projection Figures Parameters
    CUT_LEFT = 0 #N//5 # minimum is 0
    CUT_RIGHT = N #int(round(N//1.5)) # maximum is N
        
    # Catalogue Generation Parameters
    D_LOGSTART = -3
    D_LOGEND = 1
    D_BINS = 40
    MIN_NUMBER_DM_PTCS = 200
    MIN_NUMBER_STAR_PTCS = 100
    NCHAR = int(N/4)-1 
    OVERDENSITY = 200
    SAVE_FIGS = False
        
    # Morphology Parameters
    M_TOL = 1e-2
    N_WALL = 100
    N_MIN = 10
    SAFE = 0.2 # In cMpc/h
    
    # Miscallaneous
    G = 1
    FILE_TYPE = "pdf" 
    
    # Disperse
    FAST_DISPERSE = 0.25
    
    # NEXUS+
    SIG_VEC = np.logspace(-3, 5, int(200), dtype=np.float32)
    VIR_VEC = np.logspace(0, 6, int(50), dtype=np.float32)
    # At z = 0, we assume that 0.5 Mpc is the smallest scale at which we expect to find structure
    # At z = 0, We assume that 4 Mpc is the largest scale at which we expect to find structure
    RSMOOTH = np.array([np.sqrt(2)**n*0.25 for n in range(8)]).astype('float32')
    NORM = 100
    MASK = True
    CONNECTIVITY = 1
    OMEGA_M = 0.3089 # ± 0.0062, Planck2015 (lensing reconstruction + external BAO + JLA + H0), same as TNG100
    OMEGA_L = 0.6911 # ± 0.0062, same source
    
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    
    # File/Folder Locations
    SNAP_ABB = ["024","026","028","030"]
    SNAP = "024"
    VIZ_DEST =  os.path.join(currentdir, '..', 'output', 'viz')
    HDF5_SNAP_DEST = getHDF5_SNAP_DEST(SNAP)
    SHAPE_DEST = os.path.join(currentdir, '..', 'output', 'shapes')
    ALIGNMENT_DEST = os.path.join(currentdir, '..', 'output', 'alignments')
    PROJECTION_DEST = os.path.join(currentdir, '..', 'output', 'viz', 'projections')
    CAT_DEST = os.path.join(currentdir, '..', 'output', 'catalogues')
    SNAP_MAX = 16
    
def getHDF5_SNAP_DEST(snap):
    global SNAP_ABB
    return '/data/highz2/AFdata2/AF_PRL_BECDM/BBGas1024L17s14/snapdir_{0}'.format(snap)

def makeGlobalDM_TYPE(new_type, snap, start_time):
    print_status(rank, start_time, "Changing to DM type {0} and snap {1}".format(new_type.upper(), snap))
    global DM_TYPE
    global HDF5_SNAP_DEST
    global SNAP
    DM_TYPE = new_type
    HDF5_SNAP_DEST = getHDF5_SNAP_DEST(snap)
    SNAP = snap
    
def makeGlobalHALO_REGION(new_region):
    global HALO_REGION
    HALO_REGION = new_region

def makeGlobalA(new_a):
    global A
    A = new_a
    
def makeGlobalRSMOOTH(RSMOOTH_new):
    global RSMOOTH
    RSMOOTH = RSMOOTH_new
    
def getA():
    global HDF5_SNAP_DEST
    global SNAP
    f = h5py.File(r'{0}/snap_{1}.{2}.hdf5'.format(HDF5_SNAP_DEST, SNAP, 0), 'r')
    return 1/(f['Header'].attrs['Redshift']+1)