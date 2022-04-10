#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 11:05:21 2021

@author: tibor
"""

rank = 0
import config
import numpy as np
from copy import deepcopy
import nexus_plus
from plotting_utilities import plotSigProj, plotGridProjections, plotThreshold
import make_grid_cic
from get_hdf5 import getHDF5DMData
from print_msg import print_status
from memory_profiler import profile

def getDelta(): # Overdensity required for spherical virialization relative to the mean BG of the universe
    Z = config.getA()**(-1)-1
    x = (config.OMEGA_M*(1+Z)**3)/(config.OMEGA_M*(1+Z)**3+config.OMEGA_L)-1
    DELTA = (18*np.pi**2 + 82*x - 39*x**2)/(x+1) - 1 # Bryan, Norman, 1998, ApJ (Vir. Theorem + Spher. Collapse)
    return np.float32(DELTA)

@profile(precision=2)
def getSigsNexusPlus(start_time):
    """
    Create cluster, filament and wall signatures.
    Note that we assume FDM has N=1024 and WDM/CDM have N=512 resolution below."""
    
    print_status(rank, start_time, 'Starting getSigsNexusPlus() and DM_TYPE being {0}'.format(config.DM_TYPE))
    # Import hdf5 data
    dm_xyz, dm_masses, dm_smoothing, dm_velxyz = getHDF5DMData()
    del dm_smoothing; del dm_velxyz
    
    # Constructing the grid with PBC
    nexus_plus_obj = nexus_plus.NEXUS(config.N, config.L_BOX, config.RSMOOTH, config.NORM, config.SIG_VEC, config.VIR_VEC, getDelta())
    rgrid = make_grid_cic.makeGridWithCICPBC(dm_xyz[:,0].astype('float32'), dm_xyz[:,1].astype('float32'), dm_xyz[:,2].astype('float32'), dm_masses.astype('float32'), config.L_BOX, config.N)
    nexus_plus_obj.setRgrid(rgrid.astype(np.float32)) # Needed in nexus_plus_obj's methods, where rgrid is modified
    print_status(rank, start_time, "Initialized nexus_plus_obj. The smoothing scales are the following: {0}".format(config.RSMOOTH))
    print_status(rank, start_time, 'The max rgrid value is {0}. Shape of CIC rgrid is {1}. Number of NaNs in the rgrid is {2}'.format(rgrid.max(), np.shape(rgrid), np.isnan(rgrid).sum()))
    
    # Plotting rgrid projections
    plotGridProjections(rgrid)
    
    # NEXUS: Getting cluster signatures on various scales
    cluster_sig = nexus_plus_obj.getClusterSignatureVariousScales(start_time)
    print_status(rank, start_time, "Gotten cluster signatures for various scales")
    
    # Smoothing scale averaged cluster signatures at each spatial point
    cluster_sig = np.amax(cluster_sig, axis=0) # The reason why we overwrite and do not initialize another array is to save RAM
    
    # Cluster Signature Threshold Determination: Virialization method
    print_status(rank, start_time, "Calculating cluster signature threshold via virialization method")
    valid_regs = nexus_plus_obj.getFractionOfValidRegions(cluster_sig.astype(np.float32))
    s_c_cutvir = nexus_plus_obj.getVIR_VEC()[np.argmin(np.abs(valid_regs - 0.5))]
    print_status(rank, start_time, "Finished calculating cluster signature threshold: The virialization method cluster signature cut reads {0}".format(s_c_cutvir))
    
    # Clusters
    clusters = deepcopy(cluster_sig)
    clusters[clusters < s_c_cutvir] = 0.0 # Only clusters themselves are of interest, not the (overall) cluster signature per se
    
    # Plotting
    plotThreshold(nexus_plus_obj.getVIR_VEC(), valid_regs, 'cluster')
    plotSigProj(cluster_sig, "csig")
    plotSigProj(clusters, "c")
    print_status(rank, start_time, "Plotted fractions of valid clusters vs S_c and cluster signatures")    
        
    # Saving
    np.savetxt('{0}/sig_cluster_{1}_{2}.txt'.format(config.CAT_DEST, config.DM_TYPE, config.SNAP), clusters.reshape(clusters.shape[0], -1) , fmt='%1.7e')  
    print_status(rank, start_time, "Saving cluster signature under sig_cluster_{0}_{1}.txt. Max cluster signature is {2}".format(config.DM_TYPE, config.SNAP, np.max(clusters)))
    
    # Density mask on clusters
    if config.MASK == True:
        rgrid[clusters > 0.0] = 0.0
        nexus_plus_obj.setRgrid(rgrid.astype(np.float32))
        print_status(rank, start_time, "The number of points where density is set to zero is {0}".format((cluster_sig > s_c_cutvir).sum()))
    del cluster_sig
    
    
    # NEXUS: Getting filament signatures on various scales
    fil_sig = nexus_plus_obj.getFilamentSignatureVariousScales(start_time)
    print_status(rank, start_time, "Gotten filament signatures for various scales")
        
    # Smoothing scale averaged filament signatures at each spatial point
    fil_sig = np.amax(fil_sig, axis=0)
    
    # Filament Signature Threshold Determination
    print_status(rank, start_time, "Calculating filament signature threshold via mass method")
    delta_m_f_squared = nexus_plus_obj.getDeltaMSquared(fil_sig.astype(np.float32))
    s_f_cutmass = nexus_plus_obj.getSIG_VEC()[np.argmax(delta_m_f_squared)]
    print_status(rank, start_time, "Finished calculating filament signature threshold: The filament signature cut reads {0}".format(s_f_cutmass))
    
    # Filaments
    fils = deepcopy(fil_sig)
    fils[fils < s_f_cutmass] = 0.0
    fils[clusters > 0.0] = 0.0 # Fil only if not cluster too.
    
    # Plotting
    plotThreshold(nexus_plus_obj.getSIG_VEC(), delta_m_f_squared, 'filament')
    plotSigProj(fil_sig, "fsig")
    plotSigProj(fils, "f")
    print_status(rank, start_time, "Plotted DeltaMSquared vs S_f and filament signatures")    
        
    # Saving
    np.savetxt('{0}/sig_filament_{1}_{2}.txt'.format(config.CAT_DEST, config.DM_TYPE, config.SNAP), fils.reshape(fils.shape[0], -1) , fmt='%1.7e')        
    print_status(rank, start_time, "Saving filament signature under sig_filament_{0}_{1}.txt. Max filament signature is {2}".format(config.DM_TYPE, config.SNAP, np.max(fils)))
    
    # Density mask on filaments as well
    if config.MASK == True:
        rgrid[fils > 0.0] = 0.0
        nexus_plus_obj.setRgrid(rgrid.astype(np.float32))
        print_status(rank, start_time, "The number of points where density is set to zero is {0}".format((fil_sig > s_f_cutmass).sum()))
    del fil_sig; del rgrid
    
    
    # NEXUS: Getting wall signatures on various scales
    wall_sig = nexus_plus_obj.getWallSignatureVariousScales(start_time)
    print_status(rank, start_time, "Gotten wall signatures for various scales")
        
    # Smoothing scale averaged wall signatures at each spatial point
    wall_sig = np.amax(wall_sig, axis=0)
        
    # Wall Signature Threshold Determination
    print_status(rank, start_time, "Calculating wall signature threshold via mass method")
    delta_m_w_squared = nexus_plus_obj.getDeltaMSquared(wall_sig.astype(np.float32))
    s_w_cutmass = nexus_plus_obj.getSIG_VEC()[np.argmax(delta_m_w_squared)]   
    print_status(rank, start_time, "Finished calculating wall signature threshold: The wall signature cut reads {0}.".format(s_w_cutmass))
    
    # Walls
    walls = deepcopy(wall_sig)
    walls[walls < s_w_cutmass] = 0.0
    walls[clusters > 0.0] = 0.0 # Wall only if not cluster too.
    walls[fils > 0.0] = 0.0 # Wall only if not fil too.
    del clusters; del fils
    
    # Plotting
    plotThreshold(nexus_plus_obj.getSIG_VEC(), delta_m_w_squared, 'wall')
    plotSigProj(wall_sig, "wsig")
    plotSigProj(walls, "w")
    print_status(rank, start_time, "Plotted DeltaMSquared vs S_w and wall signatures")

    # Saving
    np.savetxt('{0}/sig_wall_{1}_{2}.txt'.format(config.CAT_DEST, config.DM_TYPE, config.SNAP), walls.reshape(walls.shape[0], -1) , fmt='%1.7e')  
    print_status(rank, start_time, "Saving wall signature under sig_wall_{0}_{1}.txt. Max wall signature is {2}".format(config.DM_TYPE, config.SNAP, np.max(walls)))
    
    # Clean-up
    del nexus_plus_obj; del wall_sig; del walls