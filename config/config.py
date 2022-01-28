#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 11:53:13 2020

@author: tibor
"""
    

import os
import inspect

def initialize():   
    
    # Catalogue generation
    global N # Resolution
    global L_BOX # Side of box in cMpc/h
    global DEL_X # N/L_BOX
    global FILE_TYPE # For generated viz files, note that only png and pdf are allowed
    global D_LOGSTART # Logarithm of smallest value for elliptical radius in E1 algo
    global D_BINS # Number of bins for elliptical radius discretization
    global NCHAR # The characteristic radius out to which the enclosed mass profile is calculated. <= int(N/2)-1, otherwise 2*NCHAR+1 would mean that you cover parts of box more than once
    global TOL # E1 convergence tolerance
    global N_WALL # Maximum number of iterations for convergence at given r_ell = d
    global N_MIN # Minimum number of particles (DM or star particle) in any iteration; if undercut, shape is unclassified
    global SAFE # Since E1 and S1 are sensitive to the outskirts of the object at hand (since shape tensor is not reduced), it 
    # is paramount not to leave out any outskirt particle in the outermost sphere/shell. This is achieved by adding some SAFE to max_i || COM - x_i|| to get d_max.
    global MIN_NUMBER_DM_PTCS # Minimum number of DM particles in CSH to qualify for catalogue entry
    global MIN_NUMBER_STAR_PTCS # Minimum number of star particles in galaxy to qualify for catalogue entry
    global SNAP_ABB # String (here), 3 digits after snapdir_* to differentiate snapshots from one another
    global CAT_DEST # Where to store catalogues of E1 results
    global VIZ_DEST # Where the visualizations of shapes, major axes are stored
    global HDF5_SNAP_DEST # Where we can find the snapshot
    global AXISYMMETRY_LIMIT # All A3D values above this value will lead to discarding of CSH/Gx
    global SNAP_MAX # Control the HDF5 reading of the output files
    global G # Gravitational constant for Poisson solver
    global OVERDENSITY
    
    # Analysis
    global GRID_METHOD # How to calculate grid when overplotting major axes with DM distro
    global MEAN_BINS # Number of bins for mean +- stand_dev plots
    global MEAN_BINS_M # Number of bins for mean +- stand_dev plots when we split into mass bins
    global HALO_REGION # Where to calculate the DM CSH shape; either "Full" or "Inner"
    global M_BINS_C # Number of mass bins, if const_occ is chosen
    global M_SLICES # Number of mass bins will be M_SLICES + 1, if log_split is chosen
    global M_SPLIT_TYPE # Mass splitting type, either "const_occ" or "log_slice"
    global R_LOGSTART # Logarithm of smallest value for Euclidian distance when we split into radial bins for Q, S, T averaging
    global R_BIN # Number of radial bins for Q, S, T averaging
    global T_CUT_LOW # Float, Triaxiality cut, floor
    global T_CUT_HIGH # Float, Triaxiality cut, cap
    global R_SLICES # Number of radial bins will be R_SLICES + 1, if "proper_split" is chosen, instead of "just_r_smaller_2"
    global CUT_LEFT # For projecting major axes onto CIC density plot: z_min (z-dimension)
    global CUT_RIGHT # For projecting major axes onto CIC density plot: z_max (z-dimension)
    global SHAPE_DEST # Where to store shape analysis results
    global ALIGNMENT_DEST # Where to store alignment analysis results
    global PROJECTION_DEST # Where to store projection images
            
    # Simulation Parameters
    N = 1024
    L_BOX = float(1.7) # In cMpc/h, needs to be float32 for CIC for instance, so transform when the time comes
    DEL_X = L_BOX/N
    
    # Alignment/Ellipticity Analysis Parameters
    GRID_METHOD = "CIC"
    MEAN_BINS = 7
    MEAN_BINS_M = 7
    HALO_REGION = "Full"
    M_BINS_C = 2
    M_SLICES = 1
    M_SPLIT_TYPE = "log_slice"
    R_LOGSTART = -3
    R_BIN = 10
    T_CUT_LOW = 0.0
    T_CUT_HIGH = 1.0
    R_SLICES = 0
    
    # Projection Figures Parameters
    CUT_LEFT = 0 #N//5 # minimum is 0
    CUT_RIGHT = N #int(round(N//1.5)) # maximum is N
        
    # Catalogue Generation Parameters
    D_LOGSTART = -3
    D_BINS = 40
    MIN_NUMBER_DM_PTCS = 200
    MIN_NUMBER_STAR_PTCS = 100
    NCHAR = int(N/2)-1 
    OVERDENSITY = 200
        
    # Morphology Parameters
    TOL = 1e-2
    N_WALL = 100
    N_MIN = 10
    SAFE = 0.2 # In cMpc/h
    
    # Miscallaneous
    G = 1
    FILE_TYPE = "pdf" 
    
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    
    # File/Folder Locations
    SNAP_ABB = "030"
    VIZ_DEST =  os.path.join(currentdir, '..', 'output', 'viz')
    HDF5_SNAP_DEST = getHDF5_SNAP_DEST()
    SHAPE_DEST = os.path.join(currentdir, '..', 'output', 'shapes')
    ALIGNMENT_DEST = os.path.join(currentdir, '..', 'output', 'alignments')
    PROJECTION_DEST = os.path.join(currentdir, '..', 'output', 'viz', 'projections')
    CAT_DEST = os.path.join(currentdir, '..', 'output', 'catalogues')
    SNAP_MAX = 16
    
def getHDF5_SNAP_DEST():
    global SNAP_ABB
    return '/data/highz2/AFdata2/AF_PRL_BECDM/BBGas1024L17s14/snapdir_{0}'.format(SNAP_ABB)