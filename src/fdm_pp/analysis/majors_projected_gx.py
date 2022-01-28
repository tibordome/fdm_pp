#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 12:37:48 2021

@author: tibor
"""

import numpy as np
import matplotlib.pyplot as plt
from math import isnan
import matplotlib
matplotlib.rcParams.update({'font.size': 13})
from matplotlib import colors
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import pandas as pd
import json
import make_grid_sph
import make_grid_cic
import accum
from copy import deepcopy
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
from matplotlib import cm 
from matplotlib.colors import ListedColormap
# Define top and bottom colormaps 
top = cm.get_cmap('hot', 128) # r means reversed version
bottom = cm.get_cmap('hot', 128)# combine it all
newcolors = np.vstack((bottom(np.linspace(0, 0.5, 50)),
                       top(np.linspace(0.5, 1, 128))))# create a new colormaps with a name of OrangeBlue
hot_hot = ListedColormap(newcolors, name='HotHot')
import config
from print_msg import print_status
from get_hdf5 import getHDF5Data


def projectMajorsGx(start_time):
    print_status(rank,start_time,'Starting projectMajorsGx() with snap {0}'.format(config.SNAP))
    
    if rank == 0:
    
        # Reading and Importing
        if config.HALO_REGION == "Full":
            with open('{0}/a_com_cat_overall_{1}_gx_{2}.txt'.format(config.CAT_DEST, config.DM_TYPE, config.SNAP), 'r') as filehandle:
                a_com_cat = json.load(filehandle)
            major = np.loadtxt('{0}/major_overall_{1}_gx_{2}.txt'.format(config.CAT_DEST, config.DM_TYPE, config.SNAP))
            if major.ndim == 2:
                major = major.reshape(major.shape[0], major.shape[1]//3, 3) # Has shape (number_of_gxs, 1, 3)
            else:
                if major.shape[0] == 3:
                    major = major.reshape(1, 1, 3)
            d = np.loadtxt('{0}/d_overall_{1}_gx_{2}.txt'.format(config.CAT_DEST, config.DM_TYPE, config.SNAP))
            d = d.reshape(d.shape[0], 1) # Has shape (number_of_gxs, 1)
        else:
            assert config.HALO_REGION == "Inner"
            with open('{0}/a_com_cat_local_{1}_gx_{2}.txt'.format(config.CAT_DEST, config.DM_TYPE, config.SNAP), 'r') as filehandle:
                a_com_cat = json.load(filehandle)
            major = np.loadtxt('{0}/major_local_{1}_gx_{2}.txt'.format(config.CAT_DEST, config.DM_TYPE, config.SNAP))
            if major.ndim == 2:
                major = major.reshape(major.shape[0], major.shape[1]//3, 3) # Has shape (number_of_gxs, config.D_BINS, 3)
            else:
                if major.shape[0] == (config.D_BINS+1)*3:
                    major = major.reshape(1, config.D_BINS+1, 3)
            d = np.loadtxt('{0}/d_local_{1}_gx_{2}.txt'.format(config.CAT_DEST, config.DM_TYPE, config.SNAP))
            d = d.reshape(d.shape[0], 1) # Has shape (number_of_gxs, config.D_BINS)
        nb_gxs = len(a_com_cat)
        dm_xyz, star_xyz, sh_com, nb_shs, sh_len, fof_dm_sizes, dm_masses, dm_smoothing, star_masses, star_smoothing, group_xyz, group_masses = getHDF5Data()
        group_x = group_xyz[:,0]
        group_y = group_xyz[:,1]
        group_z = group_xyz[:,2]
        dm_x = dm_xyz[:,0]
        dm_y = dm_xyz[:,1]
        dm_z = dm_xyz[:,2]
            
        print_status(rank, start_time, "The number of {0} Gxs considered is {1}".format(config.DM_TYPE.upper(), nb_gxs))
        print_status(rank, start_time, "The number of {0} halos is {1}".format(config.DM_TYPE.upper(), group_x.shape[0]))
        print_status(rank, start_time, "The smallest group mass is {0} while the largest is {1}".format(np.min(group_masses) if group_masses.shape[0] > 0 else np.nan, np.max(group_masses) if group_masses.shape[0] > 0 else np.nan))
        
        idx = np.array([int(x) for x in list(np.ones((d.shape[0],))*(-1))])
              
        # Survived gx com and masses
        major_extr = np.zeros((nb_gxs, 3))
        for gx in range(nb_gxs):
            major_extr[gx] = np.array([major[gx, idx[gx], 0], major[gx, idx[gx], 1], major[gx, idx[gx], 2]])
        gx_com = []
        for gx in range(nb_gxs):
            gx_com.append(np.array([a_com_cat[gx][3], a_com_cat[gx][4], a_com_cat[gx][5]]))
        gx_com_arr = np.array(gx_com) # Has shape (number_of_gxs, 3)
        gx_m = []
        for gx in range(nb_gxs):
            gx_m.append(a_com_cat[gx][6])
        gx_m = np.array(gx_m)
        
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
        
        # Slicing for survived gxs
        if gx_com_arr.ndim == 2:
            gx_com_x = gx_com_arr[:,0]
            gx_com_y = gx_com_arr[:,1]
            gx_com_z = gx_com_arr[:,2]
            gx_com_z_tmp = deepcopy(gx_com_z)
            gx_com_x = gx_com_x[gx_com_z_tmp > config.L_BOX*config.CUT_LEFT/config.N]
            gx_com_z_tmp = gx_com_z_tmp[gx_com_z_tmp > config.L_BOX*config.CUT_LEFT/config.N]
            gx_com_x = gx_com_x[gx_com_z_tmp < config.L_BOX*config.CUT_RIGHT/config.N]
            gx_com_z_tmp = deepcopy(gx_com_z)
            gx_com_y = gx_com_y[gx_com_z_tmp > config.L_BOX*config.CUT_LEFT/config.N]
            gx_com_z_tmp = gx_com_z_tmp[gx_com_z_tmp > config.L_BOX*config.CUT_LEFT/config.N]
            gx_com_y = gx_com_y[gx_com_z_tmp < config.L_BOX*config.CUT_RIGHT/config.N]
            gx_com_z_tmp = gx_com_z_tmp[gx_com_z_tmp < config.L_BOX*config.CUT_RIGHT/config.N]
            gx_com_arr = np.hstack((np.reshape(gx_com_x, (gx_com_x.shape[0],1)), np.reshape(gx_com_y, (gx_com_y.shape[0],1)), np.reshape(gx_com_z_tmp, (gx_com_z_tmp.shape[0],1))))
            gx_com_z_tmp = deepcopy(gx_com_z)
            gx_m = gx_m[gx_com_z_tmp > config.L_BOX*config.CUT_LEFT/config.N]
            gx_com_z_tmp = gx_com_z_tmp[gx_com_z_tmp > config.L_BOX*config.CUT_LEFT/config.N]
            gx_m = gx_m[gx_com_z_tmp < config.L_BOX*config.CUT_RIGHT/config.N]
            gx_com_z_tmp = deepcopy(gx_com_z)
            major_extr = major_extr[gx_com_z_tmp > config.L_BOX*config.CUT_LEFT/config.N]
            gx_com_z_tmp = gx_com_z_tmp[gx_com_z_tmp > config.L_BOX*config.CUT_LEFT/config.N]
            major_extr = major_extr[gx_com_z_tmp < config.L_BOX*config.CUT_RIGHT/config.N]
        else: # We have no gxs
            gx_com_arr = np.array([[np.nan, np.nan, np.nan]])
            major_extr = np.array([[np.nan, np.nan, np.nan]])
            gx_m = np.array([np.nan])
        
        # Groups, z-projection
        centers_groupsNN = np.zeros((config.N,config.N))
        stack = np.hstack((np.reshape(np.rint(group_x/config.DEL_X-0.5), (len(group_x),1)),np.reshape(np.rint(group_y/config.DEL_X-0.5), (len(group_y),1))))
        stack[stack == config.N] = config.N-1
        accummap = pd.DataFrame(stack)
        a = pd.Series(group_masses/(config.DEL_X**3)) # Mass is taken into account by NN
        centers_groupsNN += accum.accumarray(accummap, a, size=[config.N,config.N])/config.N
        centers_groupsNN = np.array(centers_groupsNN)
        centers_groupsNN[centers_groupsNN < 1e-8] = 1e-8
        
        # Define gxs, not just those that survived, z-Projection NN
        gx_cat = [[] for i in range(group_xyz.shape[0])]
        for star_ptc in range(star_xyz.shape[0]):
            dist_x = abs(star_xyz[star_ptc,0]-group_xyz[:,0])
            dist_x[dist_x > config.L_BOX/2] = config.L_BOX-dist_x[dist_x > config.L_BOX/2]
            dist_y = abs(star_xyz[star_ptc,1]-group_xyz[:,1])
            dist_y[dist_y > config.L_BOX/2] = config.L_BOX-dist_y[dist_y > config.L_BOX/2]
            dist_z = abs(star_xyz[star_ptc,2]-group_xyz[:,2])
            dist_z[dist_z > config.L_BOX/2] = config.L_BOX-dist_z[dist_z > config.L_BOX/2]
            argmin = np.argmin(dist_x**2+dist_y**2+dist_z**2)
            gx_cat[argmin].append(star_ptc) # In case star_ptc is exactly equally close to multiple subhalos, argmin will be first subhalo
        gx_cat = [x for x in gx_cat if x != []] # Remove halos with no star particles assigned
        gx_x = np.empty(0)
        gx_y = np.empty(0)
        gx_z = np.empty(0)
        gx_masses_sum = np.empty(0)
        gx_masses = np.empty(0)
        gx_com_all = np.empty(0)
        for j in range(len(gx_cat)):
            gx = np.zeros((len(gx_cat[j]),3))
            masses = np.zeros((len(gx_cat[j]),1))
            for idx, gx_ptc in enumerate(gx_cat[j]):
                gx[idx] = np.array([star_xyz[gx_ptc,0], star_xyz[gx_ptc,1], star_xyz[gx_ptc,2]])
                masses[idx] = star_masses[gx_ptc]
            gx_com_all = np.hstack((gx_com_all, np.sum(gx*np.reshape(masses, (masses.shape[0],1)), axis = 0)/masses.sum())) # COM of gx
            gx_x = np.hstack((gx_x, gx[:,0]))
            gx_y = np.hstack((gx_y, gx[:,1]))
            gx_z = np.hstack((gx_z, gx[:,2]))
            gx_masses = np.hstack((gx_masses, np.reshape(masses, (masses.shape[0],))))
            gx_masses_sum = np.hstack((gx_masses_sum, masses.sum()))
        gx_com_all = np.reshape(gx_com_all, (len(gx_cat), 3))
        
        # Slicing for all gxs
        gx_z_tmp = deepcopy(gx_z)
        gx_x = gx_x[gx_z_tmp > config.L_BOX*config.CUT_LEFT/config.N]
        gx_z_tmp = gx_z_tmp[gx_z_tmp > config.L_BOX*config.CUT_LEFT/config.N]
        gx_x = gx_x[gx_z_tmp < config.L_BOX*config.CUT_RIGHT/config.N]
        gx_z_tmp = deepcopy(gx_z)
        gx_y = gx_y[gx_z_tmp > config.L_BOX*config.CUT_LEFT/config.N]
        gx_z_tmp = gx_z_tmp[gx_z_tmp > config.L_BOX*config.CUT_LEFT/config.N]
        gx_y = gx_y[gx_z_tmp < config.L_BOX*config.CUT_RIGHT/config.N]
        gx_z_tmp = deepcopy(gx_z)
        gx_masses = gx_masses[gx_z_tmp > config.L_BOX*config.CUT_LEFT/config.N]
        gx_z_tmp = gx_z_tmp[gx_z_tmp > config.L_BOX*config.CUT_LEFT/config.N]
        gx_masses = gx_masses[gx_z_tmp < config.L_BOX*config.CUT_RIGHT/config.N]
        gx_com_z_tmp = deepcopy(gx_com_all[:,2])
        gx_masses_sum = gx_masses_sum[gx_com_z_tmp > config.L_BOX*config.CUT_LEFT/config.N]
        gx_com_z_tmp = gx_com_z_tmp[gx_com_z_tmp > config.L_BOX*config.CUT_LEFT/config.N]
        gx_masses_sum = gx_masses_sum[gx_com_z_tmp < config.L_BOX*config.CUT_RIGHT/config.N]
        
        # Gxs z-projection (not just those that survived)
        centers_NN = np.zeros((config.N,config.N))
        print_status(rank, start_time, "The number of star particles in gxs that survived slicing is {0}".format(gx_y.shape[0]))
        stack = np.hstack((np.reshape(np.rint(gx_x/config.DEL_X-0.5), (len(gx_x),1)),np.reshape(np.rint(gx_y/config.DEL_X-0.5), (len(gx_y),1))))
        stack[stack == config.N] = config.N-1
        accummap = pd.DataFrame(stack)
        a = pd.Series(gx_masses/(config.DEL_X**3))
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
        
        # Gxs only (not just those that survived)
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
        plt.title(r'z-Projected Gxs in DM')
        plt.savefig("{0}/{1}/gxs/{2}zProjectedGxsLog_{3}.pdf".format(config.PROJECTION_DEST, config.DM_TYPE, config.HALO_REGION, config.SNAP), bbox_inches="tight")
        
        # Overlaying major axes onto projected galaxies (not just survived ones) & CIC
        plt.figure()
        if gx_com_arr.shape[0] > 1 or not isnan(gx_com_arr[0,0]):
            plt.scatter(gx_com_arr[:,1], gx_com_arr[:,0], s = gx_m*10, color="lawngreen", alpha = 1)
        plt.imshow(rho_proj_cic,interpolation='None',origin='upper', extent=[0, config.L_BOX, config.L_BOX, 0], cmap=hot_hot)
        plt.gca().xaxis.set_major_locator(MultipleLocator(config.L_BOX/4))
        plt.gca().xaxis.set_minor_locator(MultipleLocator(config.L_BOX/20))
        plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%1.1f')) # integer
        plt.gca().yaxis.set_major_locator(MultipleLocator(config.L_BOX/4))
        plt.gca().yaxis.set_minor_locator(MultipleLocator(config.L_BOX/20))
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%1.1f')) # integer
        plt.xlabel(r"y (cMpc/h)", fontweight='bold')
        plt.ylabel(r"x (cMpc/h)", fontweight='bold')
        # Adding major axes of survived gxs in slice
        for gx in range(gx_com_arr.shape[0]):
            if not isnan(major_extr[gx,0]): # If major_extr is nan, do not add arrow
                norm = 1/np.sqrt(major_extr[gx,1]**2+major_extr[gx,0]**2)
                plt.annotate("", fontsize=5, xy=(gx_com_arr[gx,1]-norm*major_extr[gx,1], gx_com_arr[gx,0]-norm*major_extr[gx,0]),
                            xycoords='data', xytext=(gx_com_arr[gx,1]+norm*major_extr[gx,1], gx_com_arr[gx,0]+norm*major_extr[gx,0]),
                            textcoords='data',
                            arrowprops=dict(arrowstyle="-",
                                            linewidth = 1.,
                                            color = 'g'),
                            annotation_clip=False)
        plt.colorbar()
        plt.title(r'Gxs in DM BG')
        plt.savefig("{0}/{1}/gxs/{2}zProjGxsInDM_{3}.pdf".format(config.PROJECTION_DEST, config.DM_TYPE, config.HALO_REGION, config.SNAP), bbox_inches="tight")