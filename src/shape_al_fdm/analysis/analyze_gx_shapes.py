#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 18:11:27 2021

@author: tibor
"""

import numpy as np
from math import isnan
import matplotlib.pyplot as plt
import json
from copy import deepcopy
from splitting import M_split
from print_msg import print_status
from mpi4py import MPI
import time
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
import config
        
def analyze_gx_shapes():
    """
    Create a series of plots to analyze gx shapes"""
    
    start_time = time.time()
    print_status(rank,start_time,'Starting analyze_gx_shapes()')
    
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
        
        # Cut out last entry, since that was only for alignment purposes
        d = np.delete(d, -1, axis = 1)
        q = np.delete(q, -1, axis = 1)
        s = np.delete(s, -1, axis = 1)
        
        # M-splitting
        max_min_m, gx_m_groups, gx_com_groups, major_groups, idx_groups = M_split(gx_m, gx_com_arr, major)
         
        # Maximal elliptical radii
        d_max = np.array([np.max(d[i]) for i in range(q.shape[0])]) # Normalization differs for each gx
        R = np.logspace(config.R_LOGSTART,0,config.R_BIN)
        
        # Q
        plt.figure()
        y = [[] for i in range(config.R_BIN)]
        for gx in range(q.shape[0]):
            for rad in range(config.D_BINS):
                closest_idx = (np.abs(R - d[gx][rad]/d_max[gx])).argmin() # Determine which point in R is closest
                if isnan(q[gx][rad]):
                    continue
                else:
                    y[closest_idx].append(q[gx][rad])
        mean = np.array([np.average(z) for z in y])
        stand_err = np.array([np.std(z)/np.sqrt(len(z)) for z in y])
        plt.semilogx(R, mean, 'k-')
        plt.fill_between(R, mean-stand_err, mean+stand_err, label="FDM", edgecolor='g', alpha = 0.5)
        plt.xlabel(r"$r/r_{\mathrm{ell}}^{\mathrm{max}}$")
        plt.ylabel(r"q")
        plt.title(r"Averaged q, T: {:.2}-{:.2}".format(config.T_CUT_LOW, config.T_CUT_HIGH))
        plt.legend()
        plt.savefig("{0}/gxs/Gxq.pdf".format(config.SHAPE_DEST))   
        
        # S
        plt.figure()
        y = [[] for i in range(config.R_BIN)]
        for gx in range(s.shape[0]):
            for rad in range(config.D_BINS):
                closest_idx = (np.abs(R - d[gx][rad]/d_max[gx])).argmin() # Determine which point in R is closest
                if isnan(s[gx][rad]):
                    continue
                else:
                    y[closest_idx].append(s[gx][rad])
        mean = np.array([np.average(z) for z in y])
        stand_err = np.array([np.std(z)/np.sqrt(len(z)) for z in y])
        plt.semilogx(R, mean, 'k-')
        plt.fill_between(R, mean-stand_err, mean+stand_err, label="FDM", edgecolor='g', alpha = 0.5)
        plt.xlabel(r"$r/r_{\mathrm{ell}}^{\mathrm{max}}$")
        plt.ylabel(r"s")
        plt.title(r"Averaged s, T: {:.2}-{:.2}".format(config.T_CUT_LOW, config.T_CUT_HIGH))
        plt.legend()
        plt.savefig("{0}/gxs/Gxs.pdf".format(config.SHAPE_DEST))   
        
        # T
        plt.figure()
        T = np.zeros((q.shape[0], q.shape[1]))
        for gx in range(q.shape[0]):
            T[gx] = (1-q[gx]**2)/(1-s[gx]**2) # Triaxiality
        y = [[] for i in range(config.R_BIN)]
        for gx in range(T.shape[0]):
            for rad in range(config.D_BINS):
                closest_idx = (np.abs(R - d[gx][rad]/d_max[gx])).argmin() # Determine which point in R is closest
                if isnan(T[gx][rad]):
                    continue
                else:
                    y[closest_idx].append(T[gx][rad])
        mean = np.array([np.average(z) for z in y])
        stand_err = np.array([np.std(z)/np.sqrt(len(z)) for z in y])
        plt.semilogx(R, mean, 'k-')
        plt.fill_between(R, mean-stand_err, mean+stand_err, label="FDM", edgecolor='g', alpha = 0.5)
        plt.xlabel(r"$r/r_{\mathrm{ell}}^{\mathrm{max}}$")
        plt.ylabel(r"T")
        plt.title(r"Averaged T, T: {:.2}-{:.2}".format(config.T_CUT_LOW, config.T_CUT_HIGH))
        plt.legend()
        plt.savefig("{0}/gxs/GxT.pdf".format(config.SHAPE_DEST))   
        
        
        # Q: M-splitting
        plt.figure()
        for group in range(len(gx_m_groups)):
            y = [[] for i in range(config.R_BIN)]
            for gx in idx_groups[group]:
                for rad in range(config.D_BINS):
                    closest_idx = (np.abs(R - d[gx][rad]/d_max[gx])).argmin() # Determine which point in R is closest
                    if isnan(q[gx][rad]):
                        continue
                    else:
                        y[closest_idx].append(q[gx][rad])
            mean = np.array([np.average(z) for z in y])
            stand_err = np.array([np.std(z)/np.sqrt(len(z)) for z in y])
            
            plt.semilogx(R, mean)
            plt.fill_between(R, mean-stand_err, mean+stand_err, label = r"$FDM \ M: {:.2} - {:.2}$".format(max_min_m[group], max_min_m[group+1]), alpha = 0.5)
        plt.legend()
        plt.xlabel(r"$r/r_{\mathrm{ell}}^{\mathrm{max}}$")
        plt.ylabel(r"q")
        plt.title(r"Averaged q, T: {:.2}-{:.2}".format(config.T_CUT_LOW, config.T_CUT_HIGH))
        plt.legend()
        plt.savefig("{0}/gxs/GxqMs.pdf".format(config.SHAPE_DEST))   
        
        # S: M-splitting
        plt.figure()
        for group in range(len(gx_m_groups)):
            y = [[] for i in range(config.R_BIN)]
            for gx in idx_groups[group]:
                for rad in range(config.D_BINS):
                    closest_idx = (np.abs(R - d[gx][rad]/d_max[gx])).argmin() # Determine which point in R is closest
                    if isnan(s[gx][rad]):
                        continue
                    else:
                        y[closest_idx].append(s[gx][rad])
            mean = np.array([np.average(z) for z in y])
            stand_err = np.array([np.std(z)/np.sqrt(len(z)) for z in y])
            
            plt.semilogx(R, mean)
            plt.fill_between(R, mean-stand_err, mean+stand_err, label = r"$FDM \ M: {:.2} - {:.2}$".format(max_min_m[group], max_min_m[group+1]), alpha = 0.5)
        plt.legend()
        plt.xlabel(r"$r/r_{\mathrm{ell}}^{\mathrm{max}}$")
        plt.ylabel(r"s")
        plt.title(r"Averaged s, T: {:.2}-{:.2}".format(config.T_CUT_LOW, config.T_CUT_HIGH))
        plt.legend()
        plt.savefig("{0}/gxs/GxsMs.pdf".format(config.SHAPE_DEST))   
        
        # T: M-splitting
        plt.figure()
        for group in range(len(gx_m_groups)):
            y = [[] for i in range(config.R_BIN)]
            for gx in idx_groups[group]:
                for rad in range(config.D_BINS):
                    closest_idx = (np.abs(R - d[gx][rad]/d_max[gx])).argmin() # Determine which point in R is closest
                    if isnan(T[gx][rad]):
                        continue
                    else:
                        y[closest_idx].append(T[gx][rad])
            mean = np.array([np.average(z) for z in y])
            stand_err = np.array([np.std(z)/np.sqrt(len(z)) for z in y])
            
            plt.semilogx(R, mean)
            plt.fill_between(R, mean-stand_err, mean+stand_err, label = r"$FDM \ M: {:.2} - {:.2}$".format(max_min_m[group], max_min_m[group+1]), alpha = 0.5)
        plt.legend()
        plt.xlabel(r"$r/r_{\mathrm{ell}}^{\mathrm{max}}$")
        plt.ylabel(r"T")
        plt.title(r"Averaged T, T: {:.2}-{:.2}".format(config.T_CUT_LOW, config.T_CUT_HIGH))
        plt.legend()
        plt.savefig("{0}/gxs/GxTMs.pdf".format(config.SHAPE_DEST))   
        
        # T counting
        plt.figure()
        binwidth = 0.05
        n, bins, patches = plt.hist(x=t, bins = np.arange(np.nanmin(t), np.nanmax(t) + binwidth, binwidth), alpha=0.7, density=True, label = "CDM Gxs")
        plt.axvline(1/3, label="oblate-triaxial transition", color = "g")
        plt.axvline(2/3, label="triaxial-prolate transition", color = "r")
        plt.xlabel(r"T")
        plt.ylabel('Normalized Bin Count')
        plt.grid(axis='y', alpha=0.75)
        #plt.yscale('log', nonposy='clip') # strengthens visually the small alignment 
        plt.title(r"Triaxiality Histogram")
        plt.legend()
        plt.savefig("{0}/gxs/GxTCount.pdf".format(config.SHAPE_DEST))