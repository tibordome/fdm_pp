#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 12:37:48 2021

@author: tibor
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 13})
from matplotlib import colors
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import pandas as pd
from math import isnan
from copy import deepcopy
from matplotlib import cm 
from matplotlib.colors import ListedColormap
# Define top and bottom colormaps 
top = cm.get_cmap('hot', 128) # r means reversed version
bottom = cm.get_cmap('hot', 128)# combine it all
newcolors = np.vstack((bottom(np.linspace(0, 0.5, 50)),
                       top(np.linspace(0.5, 1, 128))))# create a new colormaps with a name of OrangeBlue
hot_hot = ListedColormap(newcolors, name='HotHot')
import make_grid_sph
import make_grid_cic
import accum
from mpi4py import MPI
from function_utilities import assembleDataDMCDMOrFDM, readDataDMCDMOrFDM
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
import config
from print_msg import print_status
from get_hdf5 import getHDF5Data

def projectMajorsHalo(start_time):
    
    print_status(rank,start_time,'Starting projectMajorsHalo() with snap {0}'.format(config.SNAP))
    
    if rank == 0:
    
        # Reading and Importing
        a_com_cat, dm_cat, d, q, s, major, r200, fof_masses_dm = readDataDMCDMOrFDM(get_skeleton = False)

        nb_halo = len(a_com_cat)
        dm_xyz, star_xyz, sh_com, nb_shs, sh_len, fof_dm_sizes, dm_masses, dm_smoothing, star_masses, star_smoothing, group_xyz, group_masses = getHDF5Data()
        group_x = group_xyz[:,0]
        group_y = group_xyz[:,1]
        group_z = group_xyz[:,2]
        dm_x = dm_xyz[:,0]
        dm_y = dm_xyz[:,1]
        dm_z = dm_xyz[:,2]
        print_status(rank, start_time, "The number of {0} halos considered (and in case of HALO_REGION == 'Inner', shape calculation converged) is {1}".format(config.DM_TYPE.upper(), nb_halo))
        print_status(rank, start_time, "The number of {0} halos is {1}".format(config.DM_TYPE.upper(), group_x.shape[0]))
        print_status(rank, start_time, "The number of {0} halos is {1}".format(config.DM_TYPE.upper(), fof_masses_dm.shape[0]))
        print_status(rank, start_time, "The smallest group mass is {0} while the largest is {1}".format(np.min(group_masses) if group_masses.shape[0] > 0 else np.nan, np.max(group_masses) if group_masses.shape[0] > 0 else np.nan))
        
        # Assembly
        halo_masses_dm, halo_com_arr, idx, major_extr, t = assembleDataDMCDMOrFDM(a_com_cat, dm_cat, d, q, s, major, r200, fof_masses_dm)
        
        # Slicing for fofs
        group_z_tmp = deepcopy(group_z)
        group_x = group_x[group_z_tmp > config.L_BOX*config.CUT_LEFT/config.N]
        group_z_tmp = group_z_tmp[group_z_tmp > config.L_BOX*config.CUT_LEFT/config.N]
        group_x = group_x[group_z_tmp < config.L_BOX*config.CUT_RIGHT/config.N]
        group_z_tmp = deepcopy(group_z)
        group_y = group_y[group_z_tmp > config.L_BOX*config.CUT_LEFT/config.N]
        group_z_tmp = group_z_tmp[group_z_tmp > config.L_BOX*config.CUT_LEFT/config.N]
        group_y = group_y[group_z_tmp < config.L_BOX*config.CUT_RIGHT/config.N]
        group_z_tmp = deepcopy(group_z)
        group_masses = group_masses[group_z_tmp > config.L_BOX*config.CUT_LEFT/config.N]
        group_z_tmp = group_z_tmp[group_z_tmp > config.L_BOX*config.CUT_LEFT/config.N]
        group_masses = group_masses[group_z_tmp < config.L_BOX*config.CUT_RIGHT/config.N]
        
        # Slicing for survived Halos: global quantities such as COM, total mass
        if halo_com_arr.ndim == 2:
            halo_com_x = halo_com_arr[:,0]
            halo_com_y = halo_com_arr[:,1]
            halo_com_z = halo_com_arr[:,2]
            halo_com_z_tmp = deepcopy(halo_com_z)
            halo_com_x = halo_com_x[halo_com_z_tmp > config.L_BOX*config.CUT_LEFT/config.N]
            halo_com_z_tmp = halo_com_z_tmp[halo_com_z_tmp > config.L_BOX*config.CUT_LEFT/config.N]
            halo_com_x = halo_com_x[halo_com_z_tmp < config.L_BOX*config.CUT_RIGHT/config.N]
            halo_com_z_tmp = deepcopy(halo_com_z)
            halo_com_y = halo_com_y[halo_com_z_tmp > config.L_BOX*config.CUT_LEFT/config.N]
            halo_com_z_tmp = halo_com_z_tmp[halo_com_z_tmp > config.L_BOX*config.CUT_LEFT/config.N]
            halo_com_y = halo_com_y[halo_com_z_tmp < config.L_BOX*config.CUT_RIGHT/config.N]
            halo_com_z_tmp = halo_com_z_tmp[halo_com_z_tmp < config.L_BOX*config.CUT_RIGHT/config.N]
            halo_com_arr = np.hstack((np.reshape(halo_com_x, (halo_com_x.shape[0],1)), np.reshape(halo_com_y, (halo_com_y.shape[0],1)), np.reshape(halo_com_z_tmp, (halo_com_z_tmp.shape[0],1))))
            halo_com_z_tmp = deepcopy(halo_com_z)
            major_extr = major_extr[halo_com_z_tmp > config.L_BOX*config.CUT_LEFT/config.N]
            halo_com_z_tmp = halo_com_z_tmp[halo_com_z_tmp > config.L_BOX*config.CUT_LEFT/config.N]
            major_extr = major_extr[halo_com_z_tmp < config.L_BOX*config.CUT_RIGHT/config.N]
            halo_com_z_tmp = deepcopy(halo_com_z)
            halo_masses_dm = halo_masses_dm[halo_com_z_tmp > config.L_BOX*config.CUT_LEFT/config.N]
            halo_com_z_tmp = halo_com_z_tmp[halo_com_z_tmp > config.L_BOX*config.CUT_LEFT/config.N]
            halo_masses_dm = halo_masses_dm[halo_com_z_tmp < config.L_BOX*config.CUT_RIGHT/config.N]
        else: # We have no halos
            halo_com_arr = np.array([[np.nan, np.nan, np.nan]])
            major_extr = np.array([[np.nan, np.nan, np.nan]])
            halo_masses_dm = np.array([np.nan])
        
        # Groups, z-projection
        centers_groupsNN = np.zeros((config.N,config.N))
        stack = np.hstack((np.reshape(np.rint(group_x/config.DEL_X-0.5), (len(group_x),1)),np.reshape(np.rint(group_y/config.DEL_X-0.5), (len(group_y),1))))
        stack[stack == config.N] = config.N-1
        accummap = pd.DataFrame(stack)
        a = pd.Series(group_masses/(config.DEL_X**3)) # Mass is taken into account by NN
        centers_groupsNN += accum.accumarray(accummap, a, size=[config.N,config.N])/config.N
        centers_groupsNN = np.array(centers_groupsNN)
        centers_groupsNN[centers_groupsNN < 1e-8] = 1e-8
        
        # Define survived halos, for z-Projection NN
        halo_x = np.empty(0)
        halo_y = np.empty(0)
        halo_z = np.empty(0)
        halo_masses = np.empty(0)
        for j in range(len(dm_cat)): # Iterate through all halos
            halo = np.zeros((len(dm_cat[j]),3))
            masses = np.zeros((len(dm_cat[j]),1))
            for idx, dm_ptc in enumerate(dm_cat[j]):
                halo[idx] = np.array([dm_xyz[dm_ptc,0], dm_xyz[dm_ptc,1], dm_xyz[dm_ptc,2]])
                masses[idx] = dm_masses[dm_ptc]
            if len(dm_cat[j]) == 0:
                continue
            halo_x = np.hstack((halo_x, halo[:,0]))
            halo_y = np.hstack((halo_y, halo[:,1]))
            halo_z = np.hstack((halo_z, halo[:,2]))
            halo_masses = np.hstack((halo_masses, np.reshape(masses, (masses.shape[0],))))
    
        # Slicing for survived Halos: DM ptc position, DM ptc mass
        halo_z_tmp = deepcopy(halo_z)
        halo_x = halo_x[halo_z_tmp > config.L_BOX*config.CUT_LEFT/config.N]
        halo_z_tmp = halo_z_tmp[halo_z_tmp > config.L_BOX*config.CUT_LEFT/config.N]
        halo_x = halo_x[halo_z_tmp < config.L_BOX*config.CUT_RIGHT/config.N]
        halo_z_tmp = deepcopy(halo_z)
        halo_y = halo_y[halo_z_tmp > config.L_BOX*config.CUT_LEFT/config.N]
        halo_z_tmp = halo_z_tmp[halo_z_tmp > config.L_BOX*config.CUT_LEFT/config.N]
        halo_y = halo_y[halo_z_tmp < config.L_BOX*config.CUT_RIGHT/config.N]
        halo_z_tmp = deepcopy(halo_z)
        halo_masses = halo_masses[halo_z_tmp > config.L_BOX*config.CUT_LEFT/config.N]
        halo_z_tmp = halo_z_tmp[halo_z_tmp > config.L_BOX*config.CUT_LEFT/config.N]
        halo_masses = halo_masses[halo_z_tmp < config.L_BOX*config.CUT_RIGHT/config.N]
        
        # Halos z-projection
        centers_NN = np.zeros((config.N,config.N))
        print_status(rank, start_time, "The number of {0} particles in halos that survived T slicing is {1}".format(config.DM_TYPE.upper(), halo_y.shape[0]))
        stack = np.hstack((np.reshape(np.rint(halo_x/config.DEL_X-0.5), (len(halo_x),1)),np.reshape(np.rint(halo_y/config.DEL_X-0.5), (len(halo_y),1))))
        stack[stack == config.N] = config.N-1
        accummap = pd.DataFrame(stack)
        a = pd.Series(halo_masses/(config.DEL_X**3))
        centers_NN += accum.accumarray(accummap, a, size=[config.N,config.N])/config.N
        plt.figure()
        centers_NN = np.array(centers_NN)
        centers_NN[centers_NN < 1e-8] = 1e-8
        
        # CIC or SPH, z-projection
        print_status(rank, start_time, "The resolution is {0}".format(config.N))
        if config.GRID_METHOD == "CIC":
            grid = make_grid_cic.makeGridWithCICPBC(dm_x.astype('float32'), dm_y.astype('float32'), dm_z.astype('float32'), dm_masses.astype('float32'), config.L_BOX, config.N)
        elif config.GRID_METHOD == "SPH":
            grid = make_grid_sph.makeGridWithSPHPBC(dm_x.astype('float32'), dm_y.astype('float32'), dm_z.astype('float32'), dm_masses.astype('float32'), np.ones_like(dm_masses)*config.L_BOX/config.N)
        print_status(rank, start_time, "Constructed the {0} grid.".format(config.GRID_METHOD))
        rho_proj_cic = np.zeros((config.N, config.N))
        for h in range(config.CUT_LEFT, config.CUT_RIGHT):
            rho_proj_cic += grid[:,:,h]
        rho_proj_cic /= config.N
        rho_proj_cic[rho_proj_cic < 1e-8] = 1e-8
        
        # Halos only
        plt.imshow(centers_NN,interpolation='None',origin='upper', extent=[0, config.L_BOX, config.L_BOX, 0], cmap = "hot", norm=colors.LogNorm(vmin=1e-8, vmax=np.max(centers_NN)))
        plt.gca().xaxis.set_major_locator(MultipleLocator(config.L_BOX/4))
        plt.gca().xaxis.set_minor_locator(MultipleLocator(config.L_BOX/20))
        plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%1.1f')) # integer
        plt.gca().yaxis.set_major_locator(MultipleLocator(config.L_BOX/4))
        plt.gca().yaxis.set_minor_locator(MultipleLocator(config.L_BOX/20))
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%1.1f')) # integer
        plt.xlabel(r"y (cMpc/h)", fontweight='bold')
        plt.ylabel(r"x (cMpc/h)", fontweight='bold')
        plt.colorbar()
        plt.title(r'z-Projected Halos')
        plt.savefig("{0}/{1}/dm/{2}zProjHalosLog_{3}.pdf".format(config.PROJECTION_DEST, config.DM_TYPE, config.HALO_REGION, config.SNAP), bbox_inches="tight")
        
        # Overlaying major axes onto projected Halos (survived ones only) & CIC
        plt.figure()
        if halo_com_arr.shape[0] > 1 or not isnan(halo_com_arr[0,0]):
            plt.scatter(halo_com_arr[:,1], halo_com_arr[:,0], s = halo_masses_dm, color="lawngreen", alpha = 1)
        plt.imshow(rho_proj_cic,interpolation='None',origin='upper', extent=[0, config.L_BOX, config.L_BOX, 0], cmap=hot_hot)
        plt.gca().xaxis.set_major_locator(MultipleLocator(config.L_BOX/4))
        plt.gca().xaxis.set_minor_locator(MultipleLocator(config.L_BOX/20))
        plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%1.1f')) # integer
        plt.gca().yaxis.set_major_locator(MultipleLocator(config.L_BOX/4))
        plt.gca().yaxis.set_minor_locator(MultipleLocator(config.L_BOX/20))
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%1.1f')) # integer
        plt.xlabel(r"y (cMpc/h)", fontweight='bold')
        plt.ylabel(r"x (cMpc/h)", fontweight='bold')
        # Adding major axes of survived halos in slice
        for halo in range(halo_com_arr.shape[0]):
            if not isnan(major_extr[halo,0]): # If major_extr is nan, do not add arrow
                norm = 1/np.sqrt(major_extr[halo,1]**2+major_extr[halo,0]**2)
                plt.annotate("", fontsize=5, xy=(halo_com_arr[halo,1]-norm*major_extr[halo,1], halo_com_arr[halo,0]-norm*major_extr[halo,0]),
                            xycoords='data', xytext=(halo_com_arr[halo,1]+norm*major_extr[halo,1], halo_com_arr[halo,0]+norm*major_extr[halo,0]),
                            textcoords='data',
                            arrowprops=dict(arrowstyle="-",
                                            linewidth = 1.,
                                            color = 'g'),
                            annotation_clip=False)
        plt.colorbar()
        plt.title(r'Halos in DM BG')
        plt.savefig("{0}/{1}/dm/{2}zProjHalosInDM_{3}.pdf".format(config.PROJECTION_DEST, config.DM_TYPE, config.HALO_REGION, config.SNAP), bbox_inches="tight")