#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 12:37:48 2021

@author: tibor
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import pandas as pd
import json
import make_grid_cic
import accum
import time
from copy import deepcopy
from matplotlib import cm 
from print_msg import print_status
from matplotlib.colors import ListedColormap
# Define top and bottom colormaps 
top = cm.get_cmap('hot', 128) # r means reversed version
bottom = cm.get_cmap('hot', 128)# combine it all
newcolors = np.vstack((bottom(np.linspace(0, 0.5, 50)),
                       top(np.linspace(0.5, 1, 128))))# create a new colormaps with a name of OrangeBlue
hot_hot = ListedColormap(newcolors, name='HotHot')
from get_hdf5 import getHDF5DMData
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
import config

def projectMajorsSH():          
    start_time = time.time()
    print_status(rank,start_time,'Starting projectMajorssh()')
    
    if rank == 0:
        
        # Import hdf5 and load catalogues
        print_status(rank,start_time,'Import hdf5 and load catalogues..')
        with open('{0}/a_com_cat_fdm_dm.txt'.format(config.CAT_DEST), 'r') as filehandle:
            a_com_cat = json.load(filehandle)
        with open('{0}/sh_cat_fdm.txt'.format(config.CAT_DEST), 'r') as filehandle:
            sh_cat = json.load(filehandle)
        sh_com = np.loadtxt('{0}/sh_coms_fdm.txt'.format(config.CAT_DEST)).astype('float32')
        nb_sh = len(a_com_cat)
        group_x = sh_com[:,0]
        group_y = sh_com[:,1]
        group_z = sh_com[:,2]
        dm_xyz, dm_masses = getHDF5DMData()  
        group_masses = np.array([np.sum(np.array([dm_masses[dm_ptc] for idx, dm_ptc in enumerate(sh_cat[sh_idx])])) for sh_idx in range(len(sh_cat))])
        dm_x = dm_xyz[:,0]
        dm_y = dm_xyz[:,1]
        dm_z = dm_xyz[:,2]
        major_load = np.loadtxt('{0}/major_fdm_dm.txt'.format(config.CAT_DEST))
        major_load = major_load.reshape(major_load.shape[0], major_load.shape[1]//3, 3) # Has shape (number_of_shs, config.D_BINS, 3)
        d = np.loadtxt('{0}/d_fdm_dm.txt'.format(config.CAT_DEST)) # Has shape (number_of_shs, config.D_BINS)
        print_status(rank, start_time, "The number of survived FDM shs considered is {0} while the number of FDM shs before A3D and minimum number cut was {1}".format(nb_sh, group_x.shape[0]))
        
        # Take half of r_ell^max as "Inner"
        if config.HALO_REGION == "Full":
            idx = np.array([int(x) for x in list(np.ones((d.shape[0],))*(-1))]).astype('int32')
        else:
            assert config.HALO_REGION == "Inner"
            idx = np.zeros((d.shape[0],), dtype = np.int32)
            for sh in range(idx.shape[0]):
                idx[sh] = np.argmin(abs(d[sh] - d[sh][-1]/2))
        
        # Survived SH com and masses
        major = np.zeros((nb_sh, 3))
        for sh in range(nb_sh):
            major[sh] = np.array([major_load[sh, idx[sh], 0], major_load[sh, idx[sh], 1], major_load[sh, idx[sh], 2]])
        sh_com = []
        for sh in range(nb_sh):
            sh_com.append(np.array([a_com_cat[sh][3], a_com_cat[sh][4], a_com_cat[sh][5]]))
        sh_com_arr = np.array(sh_com) # Has shape (number_of_shs, 3)
        sh_m = []
        for sh in range(nb_sh):
            sh_m.append(a_com_cat[sh][6])
        sh_m = np.array(sh_m)
        
        # SHs before A3D and cut
        sh_x = np.empty(0)
        sh_y = np.empty(0)
        sh_z = np.empty(0)
        sh_masses_sum = np.empty(0)
        sh_masses = np.empty(0)
        sh_com_all = np.empty(0)
        for j in range(len(sh_cat)):
            print_status(rank, start_time, "The number of DM particles in this sh is {0}".format(len(sh_cat[j])))
            sh = np.zeros((len(sh_cat[j]),3))
            masses = np.ones((len(sh_cat[j]),1))
            for idx, dm_ptc in enumerate(sh_cat[j]):
                sh[idx] = np.array([dm_xyz[dm_ptc,0], dm_xyz[dm_ptc,1], dm_xyz[dm_ptc,2]])
                masses[idx] = dm_masses[dm_ptc]
            sh_com_all = np.hstack((sh_com_all, np.sum(sh*np.reshape(masses, (masses.shape[0],1)), axis = 0)/masses.sum())) # COM of sh
            sh_x = np.hstack((sh_x, sh[:,0]))
            sh_y = np.hstack((sh_y, sh[:,1]))
            sh_z = np.hstack((sh_z, sh[:,2]))
            sh_masses = np.hstack((sh_masses, np.reshape(masses, (masses.shape[0],))))
            sh_masses_sum = np.hstack((sh_masses_sum, masses.sum()))
        sh_com_all = np.reshape(sh_com_all, (len(sh_cat), 3))
        print_status(rank, start_time, "SH masses are {0} and those survived are {1}".format(sh_masses_sum, sh_m))
    
        # Slicing
        print_status(rank, start_time, "Started slicing")
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
        
        sh_com_x = sh_com_arr[:,0]
        sh_com_y = sh_com_arr[:,1]
        sh_com_z = sh_com_arr[:,2]
        sh_com_z_tmp = deepcopy(sh_com_z)
        sh_com_x = sh_com_x[sh_com_z_tmp > config.L_BOX*config.CUT_LEFT/config.N]
        sh_com_z_tmp = sh_com_z_tmp[sh_com_z_tmp > config.L_BOX*config.CUT_LEFT/config.N]
        sh_com_x = sh_com_x[sh_com_z_tmp < config.L_BOX*config.CUT_RIGHT/config.N]
        sh_com_z_tmp = deepcopy(sh_com_z)
        sh_com_y = sh_com_y[sh_com_z_tmp > config.L_BOX*config.CUT_LEFT/config.N]
        sh_com_z_tmp = sh_com_z_tmp[sh_com_z_tmp > config.L_BOX*config.CUT_LEFT/config.N]
        sh_com_y = sh_com_y[sh_com_z_tmp < config.L_BOX*config.CUT_RIGHT/config.N]
        sh_com_z_tmp = sh_com_z_tmp[sh_com_z_tmp < config.L_BOX*config.CUT_RIGHT/config.N]
        sh_com_arr = np.hstack((np.reshape(sh_com_x, (sh_com_x.shape[0],1)), np.reshape(sh_com_y, (sh_com_y.shape[0],1)), np.reshape(sh_com_z_tmp, (sh_com_z_tmp.shape[0],1))))
        
        sh_z_tmp = deepcopy(sh_z)
        sh_x = sh_x[sh_z_tmp > config.L_BOX*config.CUT_LEFT/config.N]
        sh_z_tmp = sh_z_tmp[sh_z_tmp > config.L_BOX*config.CUT_LEFT/config.N]
        sh_x = sh_x[sh_z_tmp < config.L_BOX*config.CUT_RIGHT/config.N]
        sh_z_tmp = deepcopy(sh_z)
        sh_y = sh_y[sh_z_tmp > config.L_BOX*config.CUT_LEFT/config.N]
        sh_z_tmp = sh_z_tmp[sh_z_tmp > config.L_BOX*config.CUT_LEFT/config.N]
        sh_y = sh_y[sh_z_tmp < config.L_BOX*config.CUT_RIGHT/config.N]
        sh_z_tmp = deepcopy(sh_z)
        sh_masses = sh_masses[sh_z_tmp > config.L_BOX*config.CUT_LEFT/config.N]
        sh_z_tmp = sh_z_tmp[sh_z_tmp > config.L_BOX*config.CUT_LEFT/config.N]
        sh_masses = sh_masses[sh_z_tmp < config.L_BOX*config.CUT_RIGHT/config.N]
        sh_com_z_tmp = deepcopy(sh_com_all[:,2])
        sh_masses_sum = sh_masses_sum[sh_com_z_tmp > config.L_BOX*config.CUT_LEFT/config.N]
        sh_com_z_tmp = sh_com_z_tmp[sh_com_z_tmp > config.L_BOX*config.CUT_LEFT/config.N]
        sh_masses_sum = sh_masses_sum[sh_com_z_tmp < config.L_BOX*config.CUT_RIGHT/config.N]
        print_status(rank, start_time, "The number of DM particles in shs that survived slicing is {0}".format(sh_y.shape[0]))
        
        # Groups, z-projection
        centers_groupsNN = np.zeros((config.N,config.N))
        stack = np.hstack((np.reshape(np.rint(group_x/config.DEL_X-0.5), (len(group_x),1)),np.reshape(np.rint(group_y/config.DEL_X-0.5), (len(group_y),1))))
        stack[stack == config.N] = config.N-1
        accummap = pd.DataFrame(stack)
        a = pd.Series(group_masses/(config.DEL_X**3)) # Mass is taken into account by NN
        centers_groupsNN += accum.accumarray(accummap, a, size=[config.N,config.N])/config.N
        centers_groupsNN = np.array(centers_groupsNN)
        centers_groupsNN[centers_groupsNN < 1e-8] = 1e-8
        
        # NN rho shs, z-Projection
        centers_NN = np.zeros((config.N,config.N))
        stack = np.hstack((np.reshape(np.rint(sh_x/config.DEL_X-0.5), (len(sh_x),1)),np.reshape(np.rint(sh_y/config.DEL_X-0.5), (len(sh_y),1))))
        stack[stack == config.N] = config.N-1
        accummap = pd.DataFrame(stack)
        a = pd.Series(sh_masses/(config.DEL_X**3))
        centers_NN += accum.accumarray(accummap, a, size=[config.N,config.N])/config.N
        plt.figure()
        centers_NN = np.array(centers_NN)
        centers_NN[centers_NN < 1e-8] = 1e-8
        
        # CIC, z-projection
        print_status(rank, start_time, "Started CIC. The resolution is {0}".format(config.N))
        grid = make_grid_cic.makeGridWithCICPBC(dm_x.astype('float32'), dm_y.astype('float32'), dm_z.astype('float32'), dm_masses.astype('float32'), config.L_BOX, config.N)
        print_status(rank, start_time, "Constructed the grid")
        rho_proj_cic = np.zeros((config.N, config.N))
        for h in range(config.CUT_LEFT, config.CUT_RIGHT):
            rho_proj_cic += grid[:,:,h]
        rho_proj_cic /= (config.CUT_RIGHT - config.CUT_LEFT)
        rho_proj_cic[rho_proj_cic < 1e-8] = 1e-8

        # SHs only
        print_status(rank, start_time, "Fig creation start")
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
        plt.title(r'z-Projected SHs')
        plt.savefig("{0}/dm/{1}zProjectedshsLog.pdf".format(config.PROJECTION_DEST, config.HALO_REGION))
        
        # Overlaying major axes onto projected shs & CIC
        plt.figure()
        plt.scatter(sh_com_arr[:,1], sh_com_arr[:,0], s = sh_m*200, color="lawngreen", alpha = 1)
        plt.imshow(rho_proj_cic,interpolation='None',origin='upper', extent=[0, config.L_BOX, config.L_BOX, 0], cmap=hot_hot)
        #plt.imshow(centers_NN,interpolation='None',origin='upper', extent=[0, config.L_BOX, config.L_BOX, 0], cmap = "viridis", norm=colors.LogNorm(vmin=1e-4, vmax=np.max(centers_NN)), alpha=0.4)
        plt.gca().xaxis.set_major_locator(MultipleLocator(config.L_BOX/4))
        plt.gca().xaxis.set_minor_locator(MultipleLocator(config.L_BOX/20))
        plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%1.1f')) # integer
        plt.gca().yaxis.set_major_locator(MultipleLocator(config.L_BOX/4))
        plt.gca().yaxis.set_minor_locator(MultipleLocator(config.L_BOX/20))
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%1.1f')) # integer
        plt.xlabel(r"y (cMpc/h)", fontweight='bold')
        plt.ylabel(r"x (cMpc/h)", fontweight='bold')
        # Adding major axes of shs in slice
        for sh in range(sh_com_arr.shape[0]):
            norm = 1/np.sqrt(major[sh,1]**2+major[sh,0]**2)
            plt.annotate("", fontsize=5, xy=(sh_com_arr[sh,1]-norm*major[sh,1], sh_com_arr[sh,0]-norm*major[sh,0]),
                        xycoords='data', xytext=(sh_com_arr[sh,1]+norm*major[sh,1], sh_com_arr[sh,0]+norm*major[sh,0]),
                        textcoords='data',
                        arrowprops=dict(arrowstyle="-",
                                        linewidth = 1.,
                                        color = 'g'),
                        annotation_clip=False)
        plt.colorbar()
        plt.title(r'SHs in DM BG')
        plt.savefig("{0}/dm/{1}zProjshsInDM.pdf".format(config.PROJECTION_DEST, config.HALO_REGION))
        print_status(rank, start_time, "Finished fig creation")
   
"""
# Overlaying shs on top of CIC density projection
plt.figure()
plt.imshow(rho_proj_cic,interpolation='None',origin='upper', extent=[0, config.L_BOX, config.L_BOX, 0], cmap=hot_hot)
plt.colorbar()
plt.imshow(centers_NN,interpolation='None',origin='upper', extent=[0, config.L_BOX, config.L_BOX, 0], cmap = "hot", norm=colors.LogNorm(vmin=1e-4, vmax=np.max(centers_NN)), alpha=0.4)
plt.gca().xaxis.set_major_locator(MultipleLocator(config.L_BOX//4))
plt.gca().xaxis.set_minor_locator(MultipleLocator(config.L_BOX//20))
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%d')) # integer
plt.gca().yaxis.set_major_locator(MultipleLocator(config.L_BOX//4))
plt.gca().yaxis.set_minor_locator(MultipleLocator(config.L_BOX//20))
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%d')) # integer
plt.xlabel(r"y (cMpc/h)", fontweight='bold')
plt.ylabel(r"x (cMpc/h)", fontweight='bold')
plt.colorbar()
plt.title(r'shs in DM BG'.format(config.GRID_METHOD))
plt.savefig("zProjshsIn{0}.pdf".format(config.GRID_METHOD))


# Overlaying shs on top of CIC density projection and Halo Groups
plt.figure()
im_3 = plt.scatter(group_y, group_x, s = group_masses, color="lawngreen", alpha = 0.15)
im_2 = plt.imshow(rho_proj_cic,interpolation='None',origin='upper', extent=[0, config.L_BOX, config.L_BOX, 0], cmap=hot_hot)
im = plt.imshow(centers_NN,interpolation='None',origin='upper', extent=[0, config.L_BOX, config.L_BOX, 0], cmap = "hot", norm=colors.LogNorm(vmin=1e-4, vmax=np.max(centers_NN)), alpha=0.3) # Already set to 0 below 1e-4
plt.gca().xaxis.set_major_locator(MultipleLocator(config.L_BOX//4))
plt.gca().xaxis.set_minor_locator(MultipleLocator(config.L_BOX//20))
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%d')) # integer
plt.gca().yaxis.set_major_locator(MultipleLocator(config.L_BOX//4))
plt.gca().yaxis.set_minor_locator(MultipleLocator(config.L_BOX//20))
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%d')) # integer
plt.xlabel(r"y (cMpc/h)", fontweight='bold')
plt.ylabel(r"x (cMpc/h)", fontweight='bold')
cbar = plt.colorbar(im,fraction=0.046, pad=0.1)
cbar.set_label('shs')
cbar_2 = plt.colorbar(im_2,fraction=0.046, pad=0.13)
cbar_2.set_label('DM')
cbar_3 = plt.colorbar(im_3,fraction=0.046, pad=0.03) # Chooses viridis by default!!
cbar_3.set_label('FOF Halos')
plt.title(r'shs in DM and Halo BG'.format(config.GRID_METHOD))
plt.savefig("zProjshsIn{0}AndHalo.pdf".format(config.GRID_METHOD))

# Just Groups
plt.figure()
plt.imshow(centers_groupsNN,interpolation='None',origin='upper', cmap = "viridis", extent=[0, config.L_BOX, config.L_BOX, 0], norm=colors.LogNorm(vmin=1e-4, vmax=np.max(centers_groupsNN)))
plt.gca().xaxis.set_major_locator(MultipleLocator(config.L_BOX//4))
plt.gca().xaxis.set_minor_locator(MultipleLocator(config.L_BOX//20))
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%d')) # integer
plt.gca().yaxis.set_major_locator(MultipleLocator(config.L_BOX//4))
plt.gca().yaxis.set_minor_locator(MultipleLocator(config.L_BOX//20))
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%d')) # integer
plt.xlabel(r"y (cMpc/h)", fontweight='bold')
plt.ylabel(r"x (cMpc/h)", fontweight='bold')
plt.colorbar()
plt.title(r'Groups'.format(config.GRID_METHOD))
plt.savefig("Groups.pdf".format(config.GRID_METHOD))

# Bubble plot
plt.figure()
plt.scatter(group_y, group_x, s = group_masses, color="lawngreen")
plt.xlim(0, config.L_BOX)
plt.ylim(0, config.L_BOX)
plt.gca().set_aspect('equal', adjustable='box')
plt.gca().xaxis.set_major_locator(MultipleLocator(5))
plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%d')) # integer
plt.gca().yaxis.set_major_locator(MultipleLocator(5))
plt.gca().yaxis.set_minor_locator(MultipleLocator(1))
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%d')) # integer
plt.gca().invert_yaxis()
plt.xlabel(r"y (Mpc/h)", fontweight='bold')
plt.ylabel(r"x (Mpc/h)", fontweight='bold')
plt.title(r'z-Projected Weighted Halo Centers')
plt.savefig("zProjectedWeightedCenters.pdf")

# CIC plot
plt.figure()
plt.imshow(rho_proj_cic,interpolation='None',origin='upper', extent=[0, config.L_BOX, config.L_BOX, 0], cmap=hot_hot)#, norm=colors.LogNorm(vmin=1e-4, vmax=np.max(rho_proj_cic)))
plt.gca().xaxis.set_major_locator(MultipleLocator(config.L_BOX//4))
plt.gca().xaxis.set_minor_locator(MultipleLocator(config.L_BOX//20))
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%d')) # integer
plt.gca().yaxis.set_major_locator(MultipleLocator(config.L_BOX//4))
plt.gca().yaxis.set_minor_locator(MultipleLocator(config.L_BOX//20))
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%d')) # integer
plt.xlabel(r"y (cMpc/h)", fontweight='bold')
plt.ylabel(r"x (cMpc/h)", fontweight='bold')
plt.colorbar()
plt.title(r'z-Projected {0}-Density'.format(config.GRID_METHOD))
plt.savefig("zProjRho{0}.pdf".format(config.GRID_METHOD))

# NN plot
plt.imshow(centers_NN,interpolation='None',origin='upper', extent=[0, config.L_BOX, config.L_BOX, 0], cmap = "hot", norm=colors.LogNorm(vmin=1e-4, vmax=np.max(centers_NN)))
plt.gca().xaxis.set_major_locator(MultipleLocator(5))
plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%d')) # integer
plt.gca().yaxis.set_major_locator(MultipleLocator(5))
plt.gca().yaxis.set_minor_locator(MultipleLocator(1))
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%d')) # integer
plt.xlabel(r"y (cMpc/h)", fontweight='bold')
plt.ylabel(r"x (cMpc/h)", fontweight='bold')
plt.colorbar()
plt.title(r'z-Projected shs')
plt.savefig("zProjectedshsLog.pdf")


fig=plt.figure(1,figsize=(8,8))
ax=fig.add_subplot(111,projection='3d')
ax.scatter(dm_x,dm_y,dm_z,s=0.1)
plt.xlabel(r"x (cMpc/h)")
plt.ylabel(r"y (cMpc/h)")
ax.set_zlabel(r"z (cMpc/h)")
fig.savefig("shPointDistro.pdf")"""