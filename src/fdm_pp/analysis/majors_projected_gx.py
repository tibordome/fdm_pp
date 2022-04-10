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
import make_grid_cic
import json
import accum
from mpi4py import MPI
from function_utilities import assembleDataGx, readDataGx
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
import config
from print_msg import print_status
from get_hdf5 import getHDF5DMStarData

def projectMajorsGx(start_time):
    
    print_status(rank,start_time,'Starting projectMajorsGx() with snap {0}'.format(config.SNAP))
    
    dm_xyz, dm_masses, star_xyz, star_masses = getHDF5DMStarData()
    if rank == 0:
    
        # Reading and Importing
        a_com_cat_surv, cat_surv, d_surv, q_surv, s_surv, sh_total_mass_all = readDataGx()
        nb_gx = len(a_com_cat_surv)
        with open('{0}/gx_cat_fdm_{1}.txt'.format(config.CAT_DEST, config.SNAP), 'r') as filehandle:
            cat_all = json.load(filehandle)
        
        # Finding 
        gx_x_all = np.empty(0)
        gx_y_all = np.empty(0)
        gx_z_all = np.empty(0)
        gx_masses_all = np.empty(0)
        for j in range(len(cat_all)): # Iterate through all gxs
            gx_all = np.zeros((len(cat_all[j]),3))
            masses_all = np.zeros((len(cat_all[j]),1))
            for idx, star_ptc in enumerate(cat_all[j]):
                gx_all[idx] = np.array([star_xyz[star_ptc,0], star_xyz[star_ptc,1], star_xyz[star_ptc,2]])
                masses_all[idx] = star_masses[star_ptc]
            if len(cat_all[j]) == 0:
                continue
            gx_x_all = np.hstack((gx_x_all, gx_all[:,0]))
            gx_y_all = np.hstack((gx_y_all, gx_all[:,1]))
            gx_z_all = np.hstack((gx_z_all, gx_all[:,2]))
            gx_masses_all = np.hstack((gx_masses_all, np.reshape(masses_all, (masses_all.shape[0],))))
        
        dm_x = dm_xyz[:,0]
        dm_y = dm_xyz[:,1]
        dm_z = dm_xyz[:,2]
        print_status(rank, start_time, "The number of survived FDM gxs considered (and in case of HALO_REGION == 'Inner', gxape calculation converged) is {0}".format(nb_gx))
        print_status(rank, start_time, "The number of all FDM gxs is {0}".format(gx_x_all.gxape[0]))
        print_status(rank, start_time, "The smallest gx mass (among all gxs) is {0} while the largest is {1}".format(np.min(sh_total_mass_all) if sh_total_mass_all.gxape[0] > 0 else np.nan, np.max(sh_total_mass_all) if sh_total_mass_all.gxape[0] > 0 else np.nan))
        
        # CIC, z-projection
        print_status(rank, start_time, "The resolution is {0}".format(config.N))
        grid = make_grid_cic.makeGridWithCICPBC(dm_x.astype('float32'), dm_y.astype('float32'), dm_z.astype('float32'), dm_masses.astype('float32'), config.L_BOX, config.N)
        print_status(rank, start_time, "Constructed the {0} grid.".format(config.GRID_METHOD))
        rho_proj_cic = np.zeros((config.N, config.N))
        for h in range(config.CUT_LEFT, config.CUT_RIGHT):
            rho_proj_cic += grid[:,:,h]
        rho_proj_cic /= config.N
        rho_proj_cic[rho_proj_cic < 1e-8] = 1e-8
        
        # Slicing for all gxs
        gx_z_all_tmp = deepcopy(gx_z_all)
        gx_x_all = gx_x_all[gx_z_all_tmp > config.L_BOX*config.CUT_LEFT/config.N]
        gx_z_all_tmp = gx_z_all_tmp[gx_z_all_tmp > config.L_BOX*config.CUT_LEFT/config.N]
        gx_x_all = gx_x_all[gx_z_all_tmp < config.L_BOX*config.CUT_RIGHT/config.N]
        gx_z_all_tmp = deepcopy(gx_z_all)
        gx_y_all = gx_y_all[gx_z_all_tmp > config.L_BOX*config.CUT_LEFT/config.N]
        gx_z_all_tmp = gx_z_all_tmp[gx_z_all_tmp > config.L_BOX*config.CUT_LEFT/config.N]
        gx_y_all = gx_y_all[gx_z_all_tmp < config.L_BOX*config.CUT_RIGHT/config.N]
        gx_z_all_tmp = deepcopy(gx_z_all)
        gx_masses_all = gx_masses_all[gx_z_all_tmp > config.L_BOX*config.CUT_LEFT/config.N]
        gx_z_all_tmp = gx_z_all_tmp[gx_z_all_tmp > config.L_BOX*config.CUT_LEFT/config.N]
        gx_masses_all = gx_masses_all[gx_z_all_tmp < config.L_BOX*config.CUT_RIGHT/config.N]
        
        # All gxs, z-projection
        centers_allNN = np.zeros((config.N,config.N))
        stack = np.hstack((np.reshape(np.rint(gx_x_all/config.DEL_X-0.5), (len(gx_x_all),1)),np.reshape(np.rint(gx_y_all/config.DEL_X-0.5), (len(gx_y_all),1))))
        stack[stack == config.N] = config.N-1
        accummap = pd.DataFrame(stack)
        a = pd.Series(gx_masses_all/(config.DEL_X**3)) # Star particle mass is taken into account by NN
        centers_allNN += accum.accumarray(accummap, a, size=[config.N,config.N])/config.N
        centers_allNN = np.array(centers_allNN)
        centers_allNN[centers_allNN < 1e-8] = 1e-8
        
        # All gxs
        second_smallest = np.unique(centers_allNN)[1]
        centers_allNN[centers_allNN < second_smallest] = second_smallest
        plt.imshow(centers_allNN,interpolation='None',origin='upper', extent=[0, config.L_BOX, config.L_BOX, 0], cmap = "hot", norm=colors.LogNorm(vmin=second_smallest, vmax=np.max(centers_allNN)), alpha = 0.5)
        cbar = plt.colorbar()
        cbar.ax.get_yaxis().labelpad = 10
        cbar.ax.set_ylabel('Gxs', rotation=270)
        second_smallest = np.unique(rho_proj_cic)[1]
        rho_proj_cic[rho_proj_cic < second_smallest] = second_smallest
        plt.imshow(rho_proj_cic,interpolation='None',origin='upper', extent=[0, config.L_BOX, config.L_BOX, 0], cmap="viridis", norm=colors.LogNorm(vmin=second_smallest, vmax=np.max(rho_proj_cic)), alpha = 0.5)
        cbar = plt.colorbar()
        cbar.ax.get_yaxis().labelpad = 10
        cbar.ax.set_ylabel('DM BG', rotation=270)
        plt.gca().xaxis.set_major_locator(MultipleLocator(config.L_BOX/4))
        plt.gca().xaxis.set_minor_locator(MultipleLocator(config.L_BOX/20))
        plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%1.1f')) # integer
        plt.gca().yaxis.set_major_locator(MultipleLocator(config.L_BOX/4))
        plt.gca().yaxis.set_minor_locator(MultipleLocator(config.L_BOX/20))
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%1.1f')) # integer
        plt.xlabel(r"y (cMpc/h)", fontweight='bold')
        plt.ylabel(r"x (cMpc/h)", fontweight='bold')
        plt.title(r'z-Projected Gxs')
        plt.savefig("{0}/dm/{1}zProjAllGxsLogInDM_{2}.pdf".format(config.PROJECTION_DEST, config.HALO_REGION, config.SNAP), bbox_inches="tight")
        
        
        # Assembly of surv gxs
        sh_total_mass_surv, gx_com_arr_surv, idx_surv, major_surv, t_surv = assembleDataGx(cat_surv, a_com_cat_surv, q_surv, s_surv, sh_total_mass_all)
        
        # Slicing for surv gxs: global quantities such as COM, total mass
        if gx_com_arr_surv.ndim == 2:
            gx_com_x_surv = gx_com_arr_surv[:,0]
            gx_com_y_surv = gx_com_arr_surv[:,1]
            gx_com_z_surv = gx_com_arr_surv[:,2]
            gx_com_z_surv_tmp = deepcopy(gx_com_z_surv)
            gx_com_x_surv = gx_com_x_surv[gx_com_z_surv_tmp > config.L_BOX*config.CUT_LEFT/config.N]
            gx_com_z_surv_tmp = gx_com_z_surv_tmp[gx_com_z_surv_tmp > config.L_BOX*config.CUT_LEFT/config.N]
            gx_com_x_surv = gx_com_x_surv[gx_com_z_surv_tmp < config.L_BOX*config.CUT_RIGHT/config.N]
            gx_com_z_surv_tmp = deepcopy(gx_com_z_surv)
            gx_com_y_surv = gx_com_y_surv[gx_com_z_surv_tmp > config.L_BOX*config.CUT_LEFT/config.N]
            gx_com_z_surv_tmp = gx_com_z_surv_tmp[gx_com_z_surv_tmp > config.L_BOX*config.CUT_LEFT/config.N]
            gx_com_y_surv = gx_com_y_surv[gx_com_z_surv_tmp < config.L_BOX*config.CUT_RIGHT/config.N]
            gx_com_z_surv_tmp = gx_com_z_surv_tmp[gx_com_z_surv_tmp < config.L_BOX*config.CUT_RIGHT/config.N]
            gx_com_arr_surv = np.hstack((np.reshape(gx_com_x_surv, (gx_com_x_surv.gxape[0],1)), np.reshape(gx_com_y_surv, (gx_com_y_surv.gxape[0],1)), np.reshape(gx_com_z_surv_tmp, (gx_com_z_surv_tmp.gxape[0],1))))
            gx_com_z_surv_tmp = deepcopy(gx_com_z_surv)
            major_surv = major_surv[gx_com_z_surv_tmp > config.L_BOX*config.CUT_LEFT/config.N]
            gx_com_z_surv_tmp = gx_com_z_surv_tmp[gx_com_z_surv_tmp > config.L_BOX*config.CUT_LEFT/config.N]
            major_surv = major_surv[gx_com_z_surv_tmp < config.L_BOX*config.CUT_RIGHT/config.N]
            gx_com_z_surv_tmp = deepcopy(gx_com_z_surv)
            sh_total_mass_surv = sh_total_mass_surv[gx_com_z_surv_tmp > config.L_BOX*config.CUT_LEFT/config.N]
            gx_com_z_surv_tmp = gx_com_z_surv_tmp[gx_com_z_surv_tmp > config.L_BOX*config.CUT_LEFT/config.N]
            sh_total_mass_surv = sh_total_mass_surv[gx_com_z_surv_tmp < config.L_BOX*config.CUT_RIGHT/config.N]
        else: # We have no gxs
            gx_com_arr_surv = np.array([[np.nan, np.nan, np.nan]])
            major_surv = np.array([[np.nan, np.nan, np.nan]])
            sh_total_mass_surv = np.array([np.nan])
        
        # Define surv gxs, for z-Projection NN
        gx_x_surv = np.empty(0)
        gx_y_surv = np.empty(0)
        gx_z_surv = np.empty(0)
        gx_masses_surv = np.empty(0)
        for j in range(len(cat_surv)): # Iterate through all gxs
            gx_surv = np.zeros((len(cat_surv[j]),3))
            masses_surv = np.zeros((len(cat_surv[j]),1))
            for idx, star_ptc in enumerate(cat_surv[j]):
                gx_surv[idx] = np.array([star_xyz[star_ptc,0], star_xyz[star_ptc,1], star_xyz[star_ptc,2]])
                masses_surv[idx] = star_masses[star_ptc]
            if len(cat_surv[j]) == 0:
                continue
            gx_x_surv = np.hstack((gx_x_surv, gx_surv[:,0]))
            gx_y_surv = np.hstack((gx_y_surv, gx_surv[:,1]))
            gx_z_surv = np.hstack((gx_z_surv, gx_surv[:,2]))
            gx_masses_surv = np.hstack((gx_masses_surv, np.reshape(masses_surv, (masses_surv.gxape[0],))))
    
        # Slicing for surv gxs: star ptc position, star ptc mass
        gx_z_surv_tmp = deepcopy(gx_z_surv)
        gx_x_surv = gx_x_surv[gx_z_surv_tmp > config.L_BOX*config.CUT_LEFT/config.N]
        gx_z_surv_tmp = gx_z_surv_tmp[gx_z_surv_tmp > config.L_BOX*config.CUT_LEFT/config.N]
        gx_x_surv = gx_x_surv[gx_z_surv_tmp < config.L_BOX*config.CUT_RIGHT/config.N]
        gx_z_surv_tmp = deepcopy(gx_z_surv)
        gx_y_surv = gx_y_surv[gx_z_surv_tmp > config.L_BOX*config.CUT_LEFT/config.N]
        gx_z_surv_tmp = gx_z_surv_tmp[gx_z_surv_tmp > config.L_BOX*config.CUT_LEFT/config.N]
        gx_y_surv = gx_y_surv[gx_z_surv_tmp < config.L_BOX*config.CUT_RIGHT/config.N]
        gx_z_surv_tmp = deepcopy(gx_z_surv)
        gx_masses_surv = gx_masses_surv[gx_z_surv_tmp > config.L_BOX*config.CUT_LEFT/config.N]
        gx_z_surv_tmp = gx_z_surv_tmp[gx_z_surv_tmp > config.L_BOX*config.CUT_LEFT/config.N]
        gx_masses_surv = gx_masses_surv[gx_z_surv_tmp < config.L_BOX*config.CUT_RIGHT/config.N]
        
        # Surv gxs, z-projection
        centers_NN = np.zeros((config.N,config.N))
        print_status(rank, start_time, "The number of FDM particles in gxs that survived T slicing is {0}".format(gx_y_surv.gxape[0]))
        stack = np.hstack((np.reshape(np.rint(gx_x_surv/config.DEL_X-0.5), (len(gx_x_surv),1)),np.reshape(np.rint(gx_y_surv/config.DEL_X-0.5), (len(gx_y_surv),1))))
        stack[stack == config.N] = config.N-1
        accummap = pd.DataFrame(stack)
        a = pd.Series(gx_masses_surv/(config.DEL_X**3))
        centers_NN += accum.accumarray(accummap, a, size=[config.N,config.N])/config.N
        plt.figure()
        centers_NN = np.array(centers_NN)
        centers_NN[centers_NN < 1e-8] = 1e-8
        
        # Surv gxs only
        second_smallest = np.unique(centers_NN)[1]
        centers_NN[centers_NN < second_smallest] = second_smallest
        plt.imshow(centers_NN,interpolation='None',origin='upper', extent=[0, config.L_BOX, config.L_BOX, 0], cmap = "hot", norm=colors.LogNorm(vmin=second_smallest, vmax=np.max(centers_NN)))
        cbar = plt.colorbar()
        cbar.ax.get_yaxis().labelpad = 10
        cbar.ax.set_ylabel('Gxs', rotation=270)
        plt.gca().xaxis.set_major_locator(MultipleLocator(config.L_BOX/4))
        plt.gca().xaxis.set_minor_locator(MultipleLocator(config.L_BOX/20))
        plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%1.1f')) # integer
        plt.gca().yaxis.set_major_locator(MultipleLocator(config.L_BOX/4))
        plt.gca().yaxis.set_minor_locator(MultipleLocator(config.L_BOX/20))
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%1.1f')) # integer
        plt.xlabel(r"y (cMpc/h)", fontweight='bold')
        plt.ylabel(r"x (cMpc/h)", fontweight='bold')
        plt.title(r'z-Projected Gxs')
        plt.savefig("{0}/dm/{1}zProjGxsLog_{2}.pdf".format(config.PROJECTION_DEST, config.HALO_REGION, config.SNAP), bbox_inches="tight")
        
        # Overlaying major axes onto projected Gxs (surv ones only) & CIC
        plt.figure()
        if gx_com_arr_surv.gxape[0] > 1 or not isnan(gx_com_arr_surv[0,0]):
            plt.scatter(gx_com_arr_surv[:,1], gx_com_arr_surv[:,0], s = sh_total_mass_surv, color="lawngreen", alpha = 1)
        second_smallest = np.unique(rho_proj_cic)[0]
        plt.imshow(rho_proj_cic,interpolation='None',origin='upper', extent=[0, config.L_BOX, config.L_BOX, 0], cmap=hot_hot, norm=colors.LogNorm(vmin=second_smallest, vmax=np.max(rho_proj_cic)))
        cbar = plt.colorbar()
        cbar.ax.get_yaxis().labelpad = 10
        cbar.ax.set_ylabel('DM BG', rotation=270)
        plt.gca().xaxis.set_major_locator(MultipleLocator(config.L_BOX/4))
        plt.gca().xaxis.set_minor_locator(MultipleLocator(config.L_BOX/20))
        plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%1.1f')) # integer
        plt.gca().yaxis.set_major_locator(MultipleLocator(config.L_BOX/4))
        plt.gca().yaxis.set_minor_locator(MultipleLocator(config.L_BOX/20))
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%1.1f')) # integer
        plt.xlabel(r"y (cMpc/h)", fontweight='bold')
        plt.ylabel(r"x (cMpc/h)", fontweight='bold')
        # Adding major axes of surv gxs in slice
        for gx in range(gx_com_arr_surv.gxape[0]):
            if not isnan(major_surv[gx,0]): # If major_extr is nan, do not add arrow
                norm = 1/np.sqrt(major_surv[gx,1]**2+major_surv[gx,0]**2)
                plt.annotate("", fontsize=5, xy=(gx_com_arr_surv[gx,1]-norm*major_surv[gx,1], gx_com_arr_surv[gx,0]-norm*major_surv[gx,0]),
                            xycoords='data', xytext=(gx_com_arr_surv[gx,1]+norm*major_surv[gx,1], gx_com_arr_surv[gx,0]+norm*major_surv[gx,0]),
                            textcoords='data',
                            arrowprops=dict(arrowstyle="-",
                                            linewidth = 1.,
                                            color = 'g'),
                            annotation_clip=False)
        plt.title(r'Gxs in DM BG')
        plt.savefig("{0}/dm/{1}zProjGxsInDM_{2}.pdf".format(config.PROJECTION_DEST, config.HALO_REGION, config.SNAP), bbox_inches="tight")