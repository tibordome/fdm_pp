#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 09:54:40 2021

@author: tibor
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from copy import deepcopy
from splitting import R_split, M_split
from print_msg import print_status
import time
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
import config
        
def analyze_gx_alignments():
    """
    Create a series of plots to analyze gx alignments"""
    
    start_time = time.time()
    print_status(rank,start_time,'Starting analyze_gx_alignments()')
    
    if rank == 0:
        # Reading
        with open('{0}/a_com_cat_fdm_gx.txt'.format(config.CAT_DEST), 'r') as filehandle:
            a_com_cat = json.load(filehandle)
        print_status(rank, start_time, "The purported number of FDM Gxs considered is {0}".format(len(a_com_cat)))
        d = np.loadtxt('{0}/d_fdm_gx.txt'.format(config.CAT_DEST)) # Has shape (number_of_gxs, config.D_BINS)
        q = np.loadtxt('{0}/q_fdm_gx.txt'.format(config.CAT_DEST))
        s = np.loadtxt('{0}/s_fdm_gx.txt'.format(config.CAT_DEST))
        
        major = np.zeros((len(a_com_cat), 3))
        for gx in range(len(a_com_cat)):
            major[gx] = np.array([a_com_cat[gx][0], a_com_cat[gx][1], a_com_cat[gx][2]])
        gx_com = []
        for gx in range(len(a_com_cat)):
            gx_com.append(np.array([a_com_cat[gx][3], a_com_cat[gx][4], a_com_cat[gx][5]]))
        gx_com_arr = np.array(gx_com) # Has shape (number_of_gxs, 3)
        gx_m = []
        for gx in range(len(a_com_cat)):
            gx_m.append(a_com_cat[gx][6])
        gx_m = np.array(gx_m)
        
        # Apply T cut
        t = (1-q[:,-1]**2)/(1-s[:,-1]**2) # Triaxiality
        
        t_tmp = deepcopy(t)
        gx_m = gx_m[t_tmp >= config.T_CUT_LOW]
        t_tmp = t_tmp[t_tmp >= config.T_CUT_LOW]
        gx_m = gx_m[t_tmp <= config.T_CUT_HIGH]
        
        t_tmp = deepcopy(t)
        gx_com_arr = gx_com_arr[t_tmp >= config.T_CUT_LOW]
        t_tmp = t_tmp[t_tmp >= config.T_CUT_LOW]
        gx_com_arr = gx_com_arr[t_tmp <= config.T_CUT_HIGH]
        
        t_tmp = deepcopy(t)
        major = major[t_tmp >= config.T_CUT_LOW]
        t_tmp = t_tmp[t_tmp >= config.T_CUT_LOW]
        major = major[t_tmp <= config.T_CUT_HIGH]
        
        t_tmp = deepcopy(t)
        d = d[t_tmp >= config.T_CUT_LOW]
        t_tmp = t_tmp[t_tmp >= config.T_CUT_LOW]
        d = d[t_tmp <= config.T_CUT_HIGH]
        
        t_tmp = deepcopy(t)
        q = q[t_tmp >= config.T_CUT_LOW]
        t_tmp = t_tmp[t_tmp >= config.T_CUT_LOW]
        q = q[t_tmp <= config.T_CUT_HIGH]
        
        t_tmp = deepcopy(t)
        s = s[t_tmp >= config.T_CUT_LOW]
        t_tmp = t_tmp[t_tmp >= config.T_CUT_LOW]
        s = s[t_tmp <= config.T_CUT_HIGH]
        
        gx_com = list(gx_com_arr)
        print_status(rank, start_time, "The actual number of FDM gxs considered is {0}".format(len(gx_com)))
        
         # M-splitting
        max_min_m, gx_m_groups, gx_com_groups, major_groups, idx_groups = M_split(gx_m, gx_com_arr, major)
        
        # Respecting PBCs
        seps_raw = [(idx, idx + idx2 + 1) for idx, a in enumerate(gx_com) for idx2, b in enumerate(gx_com[idx + 1:])]
        seps = []
        for pair in range(len(seps_raw)):
            dist_x = abs(gx_com_arr[seps_raw[pair][0], 0]-gx_com_arr[seps_raw[pair][1], 0])
            if dist_x > config.L_BOX/2:
                dist_x = config.L_BOX-dist_x
            dist_y = abs(gx_com_arr[seps_raw[pair][0], 1]-gx_com_arr[seps_raw[pair][1], 1])
            if dist_y > config.L_BOX/2:
                dist_y = config.L_BOX-dist_y
            dist_z = abs(gx_com_arr[seps_raw[pair][0], 2]-gx_com_arr[seps_raw[pair][1], 2])
            if dist_z > config.L_BOX/2:
                dist_z = config.L_BOX-dist_z
            if np.sqrt(dist_x**2+dist_y**2+dist_z**2) < config.L_BOX/2:
                seps.append((seps_raw[pair][0], seps_raw[pair][1], np.sqrt(dist_x**2+dist_y**2+dist_z**2))) 
        args_sort = np.argsort(np.array(seps)[:,2])
        seps_ordered = np.sort(np.array(seps)[:,2])
        NN_seps = []
        
        # NN analysis
        for gx in range(len(gx_com)):
            dists = []
            for run in range(len(seps)):
                if seps[run][0] == gx:
                    dists.append((seps[run][0], seps[run][1], seps[run][2]))
                if seps[run][1] == gx:
                    dists.append((seps[run][1], seps[run][0], seps[run][2]))
            if dists[np.argmin(np.array(dists)[:,2])][1] == gx:
                NN_seps.append((gx, dists[np.argmin(np.array(dists)[:,2])][0], np.array(dists)[:,2].min()))
            if dists[np.argmin(np.array(dists)[:,2])][0] == gx:
                NN_seps.append((gx, dists[np.argmin(np.array(dists)[:,2])][1], np.array(dists)[:,2].min()))
        
        # SP, Mean: solid, shading: standard error on the mean
        gx1_idx = np.array(seps)[:,0][args_sort].astype(int)
        gx2_idx = np.array(seps)[:,1][args_sort].astype(int)
        x = np.linspace(np.min(seps_ordered), np.max(seps_ordered), config.MEAN_BINS)
        y = [[] for i in range(x.shape[0])]
        y_nn = [[] for i in range(x.shape[0])]
        for pair in range(gx1_idx.shape[0]):
            closest_idx = (np.abs(x - seps_ordered[pair])).argmin() # Determine which point in x is closest
            y[closest_idx].append(abs(np.dot(major[gx1_idx[pair]], gx_com_arr[gx1_idx[pair]]- gx_com_arr[gx2_idx[pair]]))/(np.linalg.norm(gx_com_arr[gx1_idx[pair]]- gx_com_arr[gx2_idx[pair]])*np.linalg.norm(major[gx1_idx[pair]])))
            y[closest_idx].append(abs(np.dot(major[gx2_idx[pair]], gx_com_arr[gx2_idx[pair]]- gx_com_arr[gx1_idx[pair]]))/(np.linalg.norm(gx_com_arr[gx2_idx[pair]]- gx_com_arr[gx1_idx[pair]])*np.linalg.norm(major[gx2_idx[pair]])))
        for nn in range(len(NN_seps)):
            closest_idx = (np.abs(x - NN_seps[nn][2])).argmin() # Determine which point in x is closest
            y_nn[closest_idx].append(abs(np.dot(major[NN_seps[nn][0]], gx_com_arr[NN_seps[nn][0]]- gx_com_arr[NN_seps[nn][1]]))/(np.linalg.norm(gx_com_arr[NN_seps[nn][0]]- gx_com_arr[NN_seps[nn][1]])*np.linalg.norm(major[NN_seps[nn][0]])))
            y_nn[closest_idx].append(abs(np.dot(major[NN_seps[nn][1]], gx_com_arr[NN_seps[nn][1]]- gx_com_arr[NN_seps[nn][0]]))/(np.linalg.norm(gx_com_arr[NN_seps[nn][1]]- gx_com_arr[NN_seps[nn][0]])*np.linalg.norm(major[NN_seps[nn][1]])))
        mean = np.array([np.average(z) for z in y])
        stand_err = np.array([np.std(z)/np.sqrt(len(z)) for z in y])
        mean_nn = np.array([np.average(z) for z in y_nn])
        stand_err_nn = np.array([np.std(z)/np.sqrt(len(z)) for z in y_nn])
        
        plt.figure()
        plt.plot(x, mean)
        plt.fill_between(x, mean-stand_err, mean+stand_err, label = "FDM, All", alpha = 0.7)
        #plt.plot(x, mean_nn)
        #plt.fill_between(x, mean_nn-stand_err_nn, mean_nn+stand_err_nn, label = "NN", alpha = 0.7)
        plt.xlabel(r"3D pair separation [cMpc/h]")
        plt.ylabel(r"$|\cos \ \theta|$, Shape-Pos.")
        plt.title(r"Shape-Position Alignment, T: {:.2}-{:.2}".format(config.T_CUT_LOW, config.T_CUT_HIGH))
        plt.legend()
        plt.savefig("{0}/gxs/GxSPMean.pdf".format(config.ALIGNMENT_DEST))
        
        plt.figure()
        plt.plot(x, mean_nn)
        plt.fill_between(x, mean_nn-stand_err_nn, mean_nn+stand_err_nn, label = "FDM, NN", alpha = 0.7)
        plt.xlabel(r"3D pair separation [cMpc/h]")
        plt.ylabel(r"$|\cos \ \theta|$, Shape-Pos.")
        plt.title(r"Shape-Position Alignment, T: {:.2}-{:.2}".format(config.T_CUT_LOW, config.T_CUT_HIGH))
        plt.legend()
        plt.savefig("{0}/gxs/GxSPMeanNN.pdf".format(config.ALIGNMENT_DEST))
        
        # SS, Mean: solid, shading: standard error on the mean
        y = [[] for i in range(x.shape[0])]
        y_nn = [[] for i in range(x.shape[0])]
        for pair in range(gx1_idx.shape[0]):
            closest_idx = (np.abs(x - seps_ordered[pair])).argmin() # Determine which point in x is closest
            y[closest_idx].append(abs(np.dot(major[gx1_idx[pair]], major[gx2_idx[pair]]))/(np.linalg.norm(major[gx1_idx[pair]])*np.linalg.norm(major[gx2_idx[pair]])))
        for nn in range(len(NN_seps)):
            closest_idx = (np.abs(x - NN_seps[nn][2])).argmin() # Determine which point in x is closest
            y_nn[closest_idx].append(abs(np.dot(major[NN_seps[nn][0]], major[NN_seps[nn][1]]))/(np.linalg.norm(major[NN_seps[nn][0]])*np.linalg.norm(major[NN_seps[nn][1]])))
        mean = np.array([np.average(z) for z in y])
        stand_err = np.array([np.std(z)/np.sqrt(len(z)) for z in y])
        mean_nn = np.array([np.average(z) for z in y_nn])
        stand_err_nn = np.array([np.std(z)/np.sqrt(len(z)) for z in y_nn])
        
        plt.figure()
        plt.plot(x, mean)
        plt.fill_between(x, mean-stand_err, mean+stand_err, label = "FDM, All", alpha = 0.7)
        plt.xlabel(r"3D pair separation [cMpc/h]")
        plt.ylabel(r"$|\cos \ \theta|$, Shape-Shape")
        plt.title(r"Shape-Shape Alignment, T: {:.2}-{:.2}".format(config.T_CUT_LOW, config.T_CUT_HIGH))
        plt.legend()
        plt.savefig("{0}/gxs/GxSSMean.pdf".format(config.ALIGNMENT_DEST))
        
        plt.figure()
        plt.plot(x, mean_nn)
        plt.fill_between(x, mean_nn-stand_err_nn, mean_nn+stand_err_nn, label = "FDM, NN", alpha = 0.7)
        plt.xlabel(r"3D pair separation [cMpc/h]")
        plt.ylabel(r"$|\cos \ \theta|$, Shape-Shape")
        plt.title(r"Shape-Shape Alignment, T: {:.2}-{:.2}".format(config.T_CUT_LOW, config.T_CUT_HIGH))
        plt.legend()
        plt.savefig("{0}/gxs/GxSSMeanNN.pdf".format(config.ALIGNMENT_DEST))    
        
        # SS
        plt.figure()
        for group in range(len(gx_m_groups)):
            seps_group_raw = [(idx, idx + idx2 + 1) for idx, a in enumerate(list(gx_com_groups[group])) for idx2, b in enumerate(list(gx_com_groups[group])[idx + 1:])]
            seps_group = []
            for pair in range(len(seps_group_raw)):
                dist_x = abs(gx_com_groups[group][seps_group_raw[pair][0], 0]-gx_com_groups[group][seps_group_raw[pair][1], 0])
                if dist_x > config.L_BOX/2:
                    dist_x = config.L_BOX-dist_x
                dist_y = abs(gx_com_groups[group][seps_group_raw[pair][0], 1]-gx_com_groups[group][seps_group_raw[pair][1], 1])
                if dist_y > config.L_BOX/2:
                    dist_y = config.L_BOX-dist_y
                dist_z = abs(gx_com_groups[group][seps_group_raw[pair][0], 2]-gx_com_groups[group][seps_group_raw[pair][1], 2])
                if dist_z > config.L_BOX/2:
                    dist_z = config.L_BOX-dist_z
                if np.sqrt(dist_x**2+dist_y**2+dist_z**2) < config.L_BOX/2:
                    seps_group.append((seps_group_raw[pair][0], seps_group_raw[pair][1], np.sqrt(dist_x**2+dist_y**2+dist_z**2))) 
            if seps_group == []:
                continue
            args_sort = np.argsort(np.array(seps_group)[:,2])
            seps_ordered = np.sort(np.array(seps_group)[:,2])
            print_status(rank, start_time, "The number of gx pairs in group {0} is {1}".format(group+1, seps_ordered.shape[0]))
            gx1_idx = np.array(seps_group)[:,0][args_sort].astype(int)
            gx2_idx = np.array(seps_group)[:,1][args_sort].astype(int)
            
            x = np.linspace(np.min(seps_ordered), np.max(seps_ordered), config.MEAN_BINS_M)
            y = [[] for i in range(x.shape[0])]
            for pair in range(gx1_idx.shape[0]):
                closest_idx = (np.abs(x - seps_ordered[pair])).argmin() # Determine which point in x is closest
                y[closest_idx].append(abs(np.dot(major_groups[group][gx1_idx[pair]], major_groups[group][gx2_idx[pair]]))/(np.linalg.norm(major_groups[group][gx1_idx[pair]])*np.linalg.norm(major_groups[group][gx2_idx[pair]])))
            mean = np.array([np.average(z) for z in y])
            stand_err = np.array([np.std(z)/np.sqrt(len(z)) for z in y])
            
            plt.plot(x, mean)
            plt.fill_between(x, mean-stand_err, mean+stand_err, label = r"$FDM \ M: {:.2} - {:.2}$".format(max_min_m[group], max_min_m[group+1]), alpha = 0.5)
        plt.legend()
        plt.xlabel(r"3D pair separation [cMpc/h]")
        plt.ylabel(r"$|\cos \ \theta|$, Shape-Shape")
        plt.title(r"SS Alignment, T: {:.2}-{:.2}".format(config.T_CUT_LOW, config.T_CUT_HIGH))
        plt.savefig("{0}/gxs/GxSSMeanMs.pdf".format(config.ALIGNMENT_DEST))    
        
        # SP
        plt.figure()
        for group in range(len(gx_m_groups)):
            seps_group_raw = [(idx, idx + idx2 + 1) for idx, a in enumerate(list(gx_com_groups[group])) for idx2, b in enumerate(list(gx_com_groups[group])[idx + 1:])]
            seps_group = []
            for pair in range(len(seps_group_raw)):
                dist_x = abs(gx_com_groups[group][seps_group_raw[pair][0], 0]-gx_com_groups[group][seps_group_raw[pair][1], 0])
                if dist_x > config.L_BOX/2:
                    dist_x = config.L_BOX-dist_x
                dist_y = abs(gx_com_groups[group][seps_group_raw[pair][0], 1]-gx_com_groups[group][seps_group_raw[pair][1], 1])
                if dist_y > config.L_BOX/2:
                    dist_y = config.L_BOX-dist_y
                dist_z = abs(gx_com_groups[group][seps_group_raw[pair][0], 2]-gx_com_groups[group][seps_group_raw[pair][1], 2])
                if dist_z > config.L_BOX/2:
                    dist_z = config.L_BOX-dist_z
                if np.sqrt(dist_x**2+dist_y**2+dist_z**2) < config.L_BOX/2:
                    seps_group.append((seps_group_raw[pair][0], seps_group_raw[pair][1], np.sqrt(dist_x**2+dist_y**2+dist_z**2))) 
            if seps_group == []:
                continue
            args_sort = np.argsort(np.array(seps_group)[:,2])
            seps_ordered = np.sort(np.array(seps_group)[:,2])
            print_status(rank, start_time, "The number of gx pairs in group {0} is {1}".format(group+1, seps_ordered.shape[0]))
            gx1_idx = np.array(seps_group)[:,0][args_sort].astype(int)
            gx2_idx = np.array(seps_group)[:,1][args_sort].astype(int)
            
            x = np.linspace(np.min(seps_ordered), np.max(seps_ordered), config.MEAN_BINS_M)
            y = [[] for i in range(x.shape[0])]
            for pair in range(gx1_idx.shape[0]):
                closest_idx = (np.abs(x - seps_ordered[pair])).argmin() # Determine which point in x is closest
                y[closest_idx].append(abs(np.dot(major_groups[group][gx1_idx[pair]], gx_com_groups[group][gx1_idx[pair]]- gx_com_groups[group][gx2_idx[pair]]))/(np.linalg.norm(gx_com_groups[group][gx1_idx[pair]]- gx_com_groups[group][gx2_idx[pair]])*np.linalg.norm(major_groups[group][gx1_idx[pair]])))
                y[closest_idx].append(abs(np.dot(major_groups[group][gx2_idx[pair]], gx_com_groups[group][gx2_idx[pair]]- gx_com_groups[group][gx1_idx[pair]]))/(np.linalg.norm(gx_com_groups[group][gx2_idx[pair]]- gx_com_groups[group][gx1_idx[pair]])*np.linalg.norm(major_groups[group][gx2_idx[pair]])))
            mean = np.array([np.average(z) for z in y])
            stand_err = np.array([np.std(z)/np.sqrt(len(z)) for z in y])
            
            plt.plot(x, mean)
            plt.fill_between(x, mean-stand_err, mean+stand_err, label = r"$FDM \ M: {:.2} - {:.2}$".format(max_min_m[group], max_min_m[group+1]), alpha = 0.5)
        plt.legend()
        plt.xlabel(r"3D pair separation [cMpc/h]")
        plt.ylabel(r"$|\cos \ \theta|$, Shape-Position")
        plt.title(r"SP Alignment, T: {:.2}-{:.2}".format(config.T_CUT_LOW, config.T_CUT_HIGH))
        plt.savefig("{0}/gxs/GxSPMeanMs.pdf".format(config.ALIGNMENT_DEST))    
        
        # SP bin count
        plt.figure()
        # All
        args_sort = np.argsort(np.array(seps)[:,2])
        seps_ordered = np.sort(np.array(seps)[:,2])
        gx1_idx = np.array(seps)[:,0][args_sort].astype(int)
        gx2_idx = np.array(seps)[:,1][args_sort].astype(int)
        shape_pos = np.zeros((2*gx1_idx.shape[0],)) # Should have shape (number_of_pairs,)
        for pair in range(gx1_idx.shape[0]):
            shape_pos[2*pair] = abs(np.dot(major[gx1_idx[pair]], gx_com_arr[gx1_idx[pair]]- gx_com_arr[gx2_idx[pair]]))/(np.linalg.norm(gx_com_arr[gx1_idx[pair]]- gx_com_arr[gx2_idx[pair]])*np.linalg.norm(major[gx1_idx[pair]]))
            shape_pos[2*pair+1] = abs(np.dot(major[gx2_idx[pair]], gx_com_arr[gx2_idx[pair]]- gx_com_arr[gx1_idx[pair]]))/(np.linalg.norm(gx_com_arr[gx2_idx[pair]]- gx_com_arr[gx1_idx[pair]])*np.linalg.norm(major[gx2_idx[pair]]))
        n, bins, patches = plt.hist(x=shape_pos, density=True, label = "FDM, All", alpha = 0.7)
        plt.xlabel(r"$|\cos \ \theta|$, Shape-Position")
        plt.ylabel('Normalized Bin Count')
        plt.grid(axis='y', alpha=0.75)
        #plt.yscale('log', nonposy='clip') # strengthens visually the small alignment 
        plt.title(r"SP Histogram, T: {:.2}-{:.2}".format(config.T_CUT_LOW, config.T_CUT_HIGH))
        plt.legend()
        plt.savefig("{0}/gxs/GxSPCount.pdf".format(config.ALIGNMENT_DEST))    
        
        # SP bin count, NN
        plt.figure()
        sp = []
        for run in range(len(NN_seps)):
            sp.append(abs(np.dot(major[NN_seps[run][0]], gx_com_arr[NN_seps[run][0]]- gx_com_arr[NN_seps[run][1]]))/(np.linalg.norm(gx_com_arr[NN_seps[run][0]]- gx_com_arr[NN_seps[run][1]])*np.linalg.norm(major[NN_seps[run][0]]))) 
        n, bins, patches = plt.hist(x=sp, density=True, label = "FDM, NN", alpha = 0.7)
        plt.xlabel(r"$|\cos \ \theta|$, Shape-Position")
        plt.ylabel('Normalized Bin Count')
        plt.grid(axis='y', alpha=0.75)
        #plt.yscale('log', nonposy='clip') # strengthens visually the small alignment 
        plt.title(r"SP Histogram, T: {:.2}-{:.2}".format(config.T_CUT_LOW, config.T_CUT_HIGH))
        plt.legend()
        plt.savefig("{0}/gxs/GxSPCountNN.pdf".format(config.ALIGNMENT_DEST))    
        
        # SS bin count
        plt.figure()
        # All
        args_sort = np.argsort(np.array(seps)[:,2])
        seps_ordered = np.sort(np.array(seps)[:,2])
        gx1_idx = np.array(seps)[:,0][args_sort].astype(int)
        gx2_idx = np.array(seps)[:,1][args_sort].astype(int)
        shape_shape = np.zeros((gx1_idx.shape[0],)) # Should have shape (number_of_pairs,)
        for pair in range(gx1_idx.shape[0]):
            shape_shape[pair] = abs(np.dot(major[gx1_idx[pair]], major[gx2_idx[pair]]))/(np.linalg.norm(major[gx1_idx[pair]])*np.linalg.norm(major[gx2_idx[pair]]))
        n, bins, patches = plt.hist(x=shape_shape, density=True, label = "FDM, All", alpha = 0.7)
        plt.xlabel(r"$|\cos \ \theta|$, Shape-Shape")
        plt.ylabel('Normalized Bin Count')
        plt.grid(axis='y', alpha=0.75)
        #plt.yscale('log', nonposy='clip') # strengthens visually the small alignment 
        plt.title(r"SS Histogram, T: {:.2}-{:.2}".format(config.T_CUT_LOW, config.T_CUT_HIGH))
        plt.legend()
        plt.savefig("{0}/gxs/GxSSCount.pdf".format(config.ALIGNMENT_DEST))    
        
        # SS bin count, NN
        plt.figure()
        ss = []
        for run in range(len(NN_seps)):
            ss.append(abs(np.dot(major[NN_seps[run][0]], major[NN_seps[run][1]]))/(np.linalg.norm(major[NN_seps[run][0]])*np.linalg.norm(major[NN_seps[run][1]]))) 
        n, bins, patches = plt.hist(x=ss, density=True, label = "FDM, NN", alpha = 0.7)
        plt.xlabel(r"$|\cos \ \theta|$, Shape-Shape")
        plt.ylabel('Normalized Bin Count')
        plt.grid(axis='y', alpha=0.75)
        #plt.yscale('log', nonposy='clip') # strengthens visually the small alignment 
        plt.title(r"SS Histogram, T: {:.2}-{:.2}".format(config.T_CUT_LOW, config.T_CUT_HIGH))
        plt.legend()
        plt.savefig("{0}/gxs/GxSSCountNN.pdf".format(config.ALIGNMENT_DEST))    
            
        # SP bin count, r splitting
        plt.figure()
        max_min_r, number_of_groups = R_split(seps)  
        for group in range(number_of_groups):
            sp = []
            for run in range(len(seps)):
                if seps[run][2] >= max_min_r[group] and seps[run][2] <= max_min_r[group+1]:
                    sp.append(abs(np.dot(major[seps[run][0]], gx_com_arr[seps[run][0]]- gx_com_arr[seps[run][1]]))/(np.linalg.norm(gx_com_arr[seps[run][0]]- gx_com_arr[seps[run][1]])*np.linalg.norm(major[seps[run][0]]))) 
                    sp.append(abs(np.dot(major[seps[run][1]], gx_com_arr[seps[run][1]]- gx_com_arr[seps[run][0]]))/(np.linalg.norm(gx_com_arr[seps[run][1]]- gx_com_arr[seps[run][0]])*np.linalg.norm(major[seps[run][1]]))) 
            n, bins, patches = plt.hist(x=sp, alpha=0.7, density=True, label = "FDM, All, r: {:.2} - {:.2}".format(max_min_r[group], max_min_r[group+1]))
        plt.xlabel(r"$|\cos \ \theta|$, Shape-Pos")
        plt.ylabel('Normalized Bin Count')
        plt.grid(axis='y', alpha=0.75)
        #plt.yscale('log', nonposy='clip') # strengthens visually the small alignment 
        plt.title(r"SP Histogram, T: {:.2}-{:.2}".format(config.T_CUT_LOW, config.T_CUT_HIGH))
        plt.legend()
        plt.savefig("{0}/gxs/GxSPCountRSmaller2.pdf".format(config.ALIGNMENT_DEST))    
        
        # SP bin count, r splitting, NN
        plt.figure()
        max_min_r, number_of_groups = R_split(seps)  
        for group in range(number_of_groups):
            sp = []
            for run in range(len(NN_seps)):
                if NN_seps[run][2] >= max_min_r[group] and NN_seps[run][2] <= max_min_r[group+1]:
                    sp.append(abs(np.dot(major[NN_seps[run][0]], gx_com_arr[NN_seps[run][0]]- gx_com_arr[NN_seps[run][1]]))/(np.linalg.norm(gx_com_arr[NN_seps[run][0]]- gx_com_arr[NN_seps[run][1]])*np.linalg.norm(major[NN_seps[run][0]]))) 
                    sp.append(abs(np.dot(major[NN_seps[run][1]], gx_com_arr[NN_seps[run][1]]- gx_com_arr[NN_seps[run][0]]))/(np.linalg.norm(gx_com_arr[NN_seps[run][1]]- gx_com_arr[NN_seps[run][0]])*np.linalg.norm(major[NN_seps[run][1]]))) 
            n, bins, patches = plt.hist(x=sp, alpha=0.7, density=True, label = "FDM, NN, r: {:.2} - {:.2}".format(max_min_r[group], max_min_r[group+1]))
        plt.xlabel(r"$|\cos \ \theta|$, Shape-Pos")
        plt.ylabel('Normalized Bin Count')
        plt.grid(axis='y', alpha=0.75)
        #plt.yscale('log', nonposy='clip') # strengthens visually the small alignment 
        plt.title(r"SP Histogram, T: {:.2}-{:.2}".format(config.T_CUT_LOW, config.T_CUT_HIGH))
        plt.legend()
        plt.savefig("{0}/gxs/GxSPCountRSmaller2NN.pdf".format(config.ALIGNMENT_DEST))    
        
        # SS bin count, r splitting
        plt.figure()
        max_min_r, number_of_groups = R_split(seps)  
        for group in range(number_of_groups):
            ss = []
            for run in range(len(seps)):
                if seps[run][2] >= max_min_r[group] and seps[run][2] <= max_min_r[group+1]:
                    ss.append(abs(np.dot(major[seps[run][0]], major[seps[run][1]]))/(np.linalg.norm(major[seps[run][0]])*np.linalg.norm(major[seps[run][1]]))) 
            n, bins, patches = plt.hist(x=ss, alpha=0.7, density=True, label = "FDM, All, r: {:.2} - {:.2}".format(max_min_r[group], max_min_r[group+1]))
        plt.xlabel(r"$|\cos \ \theta|$, Shape-Shape")
        plt.ylabel('Normalized Bin Count')
        plt.grid(axis='y', alpha=0.75)
        #plt.yscale('log', nonposy='clip') # strengthens visually the small alignment 
        plt.title(r"SS Histogram, T: {:.2}-{:.2}".format(config.T_CUT_LOW, config.T_CUT_HIGH))
        plt.legend()
        plt.savefig("{0}/gxs/GxSSCountRSmaller2.pdf".format(config.ALIGNMENT_DEST))    
        
        # SS bin count, r splitting, NN
        plt.figure()
        max_min_r, number_of_groups = R_split(seps)  
        for group in range(number_of_groups):
            ss = []
            for run in range(len(NN_seps)):
                if NN_seps[run][2] >= max_min_r[group] and NN_seps[run][2] <= max_min_r[group+1]:
                    ss.append(abs(np.dot(major[NN_seps[run][0]], major[NN_seps[run][1]]))/(np.linalg.norm(major[NN_seps[run][0]])*np.linalg.norm(major[NN_seps[run][1]]))) 
            n, bins, patches = plt.hist(x=ss, alpha=0.7, density=True, label = "FDM, NN, r: {:.2} - {:.2}".format(max_min_r[group], max_min_r[group+1]))
        plt.xlabel(r"$|\cos \ \theta|$, Shape-Shape")
        plt.ylabel('Normalized Bin Count')
        plt.grid(axis='y', alpha=0.75)
        #plt.yscale('log', nonposy='clip') # strengthens visually the small alignment 
        plt.title(r"SS Histogram, T: {:.2}-{:.2}".format(config.T_CUT_LOW, config.T_CUT_HIGH))
        plt.legend()
        plt.savefig("{0}/gxs/GxSSCountRSmaller2NN.pdf".format(config.ALIGNMENT_DEST))