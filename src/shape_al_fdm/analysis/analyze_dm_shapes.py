#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 09:54:40 2021

@author: tibor
"""

import numpy as np
from math import isnan
import matplotlib.pyplot as plt
import json
from copy import deepcopy
from splitting import M_split
from print_msg import print_status
import time
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
import config

def analyze_dm_shapes():
    """
    Create a series of plots to analyze CSH shapes"""
    
    start_time = time.time()
    print_status(rank,start_time,'Starting analyze_dm_shapes()')    
    
    if rank == 0:
        # Reading
        with open('{0}/a_com_cat_fdm_dm.txt'.format(config.CAT_DEST), 'r') as filehandle:
            a_com_cat = json.load(filehandle)
        print_status(rank, start_time, "The purported number of FDM SHs considered is {0}".format(len(a_com_cat)))
        d = np.loadtxt('{0}/d_fdm_dm.txt'.format(config.CAT_DEST))
        q = np.loadtxt('{0}/q_fdm_dm.txt'.format(config.CAT_DEST))
        s = np.loadtxt('{0}/s_fdm_dm.txt'.format(config.CAT_DEST))
        
        sh_com = []
        for sh in range(len(a_com_cat)):
            sh_com.append(np.array([a_com_cat[sh][3], a_com_cat[sh][4], a_com_cat[sh][5]]))
        sh_com_arr = np.array(sh_com) # Has shape (number_of_shs, 3)
        sh_m = []
        for sh in range(len(a_com_cat)):
            sh_m.append(a_com_cat[sh][6])
        sh_m = np.array(sh_m)
        
        # Cut out last entry, since that was only for alignment purposes
        d_tmp = np.delete(d, -1, axis = 1)
        # Take half of r_ell^max as "Inner"
        if config.HALO_REGION == "Full":
            idx = np.array([int(x) for x in list(np.ones((d.shape[0],))*(-1))])
        else:
            assert config.HALO_REGION == "Inner"
            idx = np.zeros((d.shape[0],), dtype = np.int32)
            for sh in range(idx.shape[0]):
                idx[sh] = np.argmin(abs(d[sh] - d_tmp[sh][-1]/2))
            
        # Apply T cut
        t = np.zeros((d.shape[0],))
        for sh in range(idx.shape[0]):
            t[sh] = (1-q[sh,idx[sh]]**2)/(1-s[sh,idx[sh]]**2) # Triaxiality
        
        t = np.nan_to_num(t)
        t_tmp = deepcopy(t)
        sh_m = sh_m[t_tmp >= config.T_CUT_LOW]
        t_tmp = t_tmp[t_tmp >= config.T_CUT_LOW]
        sh_m = sh_m[t_tmp <= config.T_CUT_HIGH]
        
        t_tmp = deepcopy(t)
        sh_com_arr = sh_com_arr[t_tmp >= config.T_CUT_LOW]
        t_tmp = t_tmp[t_tmp >= config.T_CUT_LOW]
        sh_com_arr = sh_com_arr[t_tmp <= config.T_CUT_HIGH]
        
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
        
        sh_com = list(sh_com_arr)
        print_status(rank, start_time, "The actual number of FDM SHs considered is {0}".format(len(sh_com)))
        
        # M-splitting
        max_min_m, sh_m_groups, sh_com_groups, major_groups, idx_groups = M_split(sh_m, sh_com_arr)   
        
        d_max = np.array([np.max(d[i]) for i in range(q.shape[0])]) # Normalization differs for each sh
        R = np.logspace(config.R_LOGSTART,0,config.R_BIN)
        
        # Q
        plt.figure()
        y = [[] for i in range(config.R_BIN)]
        for sh in range(q.shape[0]):
            for rad in range(config.D_BINS):
                closest_idx = (np.abs(R - d[sh][rad]/d_max[sh])).argmin() # Determine which point in R is closest
                if isnan(q[sh][rad]):
                    continue
                else:
                    y[closest_idx].append(q[sh][rad])
        mean = np.array([np.average(z) for z in y])
        stand_err = np.array([np.std(z)/np.sqrt(len(z)) for z in y])
        plt.semilogx(R, mean, 'k-')
        plt.fill_between(R, mean-stand_err, mean+stand_err, label="FDM", edgecolor='g', alpha = 0.5)
        plt.xlabel(r"$r/r_{\mathrm{ell}}^{\mathrm{max}}$")
        plt.ylabel(r"q")
        plt.title(r"Averaged q, T: {:.2}-{:.2}".format(config.T_CUT_LOW, config.T_CUT_HIGH))
        plt.legend()
        plt.savefig("{0}/dm/CSHq.pdf".format(config.SHAPE_DEST))
        
        # S
        plt.figure()
        y = [[] for i in range(config.R_BIN)]
        for sh in range(s.shape[0]):
            for rad in range(config.D_BINS):
                closest_idx = (np.abs(R - d[sh][rad]/d_max[sh])).argmin() # Determine which point in R is closest
                if isnan(s[sh][rad]):
                    continue
                else:
                    y[closest_idx].append(s[sh][rad])
        mean = np.array([np.average(z) for z in y])
        stand_err = np.array([np.std(z)/np.sqrt(len(z)) for z in y])
        plt.semilogx(R, mean, 'k-')
        plt.fill_between(R, mean-stand_err, mean+stand_err, label="FDM", edgecolor='g', alpha = 0.5)
        
        plt.xlabel(r"$r/r_{\mathrm{ell}}^{\mathrm{max}}$")
        plt.ylabel(r"s")
        plt.title(r"Averaged s, T: {:.2}-{:.2}".format(config.T_CUT_LOW, config.T_CUT_HIGH))
        plt.legend()
        plt.savefig("{0}/dm/SHs.pdf".format(config.SHAPE_DEST))
        
        # T
        plt.figure()
        T = np.zeros((q.shape[0], q.shape[1]))
        for sh in range(q.shape[0]):
            T[sh] = (1-q[sh]**2)/(1-s[sh]**2) # Triaxiality
        y = [[] for i in range(config.R_BIN)]
        for sh in range(T.shape[0]):
            for rad in range(config.D_BINS):
                closest_idx = (np.abs(R - d[sh][rad]/d_max[sh])).argmin() # Determine which point in R is closest
                if isnan(T[sh][rad]):
                    continue
                else:
                    y[closest_idx].append(T[sh][rad])
        mean = np.array([np.average(z) for z in y])
        stand_err = np.array([np.std(z)/np.sqrt(len(z)) for z in y])
        plt.semilogx(R, mean, 'k-')
        plt.fill_between(R, mean-stand_err, mean+stand_err, label="FDM", edgecolor='g', alpha = 0.5)
        
        plt.xlabel(r"$r/r_{\mathrm{ell}}^{\mathrm{max}}$")
        plt.ylabel(r"T")
        plt.title(r"Averaged T, T: {:.2}-{:.2}".format(config.T_CUT_LOW, config.T_CUT_HIGH))
        plt.legend()
        plt.savefig("{0}/dm/CSHT.pdf".format(config.SHAPE_DEST))
        
        
        # Q: M-splitting
        plt.figure()
        for group in range(len(sh_m_groups)):
            y = [[] for i in range(config.R_BIN)]
            for sh in idx_groups[group]:
                for rad in range(config.D_BINS):
                    closest_idx = (np.abs(R - d[sh][rad]/d_max[sh])).argmin() # Determine which point in R is closest
                    if isnan(q[sh][rad]):
                        continue
                    else:
                        y[closest_idx].append(q[sh][rad])
            mean = np.array([np.average(z) for z in y])
            stand_err = np.array([np.std(z)/np.sqrt(len(z)) for z in y])
            
            plt.semilogx(R, mean)
            plt.fill_between(R, mean-stand_err, mean+stand_err, label = r"$FDM \ M: {:.2} - {:.2}$".format(max_min_m[group], max_min_m[group+1]), alpha = 0.5)
        plt.legend()
        plt.xlabel(r"$r/r_{\mathrm{ell}}^{\mathrm{max}}$")
        plt.ylabel(r"q")
        plt.title(r"Averaged q, T: {:.2}-{:.2}".format(config.T_CUT_LOW, config.T_CUT_HIGH))
        plt.legend()
        plt.savefig("{0}/dm/CSHqMs.pdf".format(config.SHAPE_DEST))
        
        # S: M-splitting
        plt.figure()
        for group in range(len(sh_m_groups)):
            y = [[] for i in range(config.R_BIN)]
            for sh in idx_groups[group]:
                for rad in range(config.D_BINS):
                    closest_idx = (np.abs(R - d[sh][rad]/d_max[sh])).argmin() # Determine which point in R is closest
                    if isnan(s[sh][rad]):
                        continue
                    else:
                        y[closest_idx].append(s[sh][rad])
            mean = np.array([np.average(z) for z in y])
            stand_err = np.array([np.std(z)/np.sqrt(len(z)) for z in y])
            
            plt.semilogx(R, mean)
            plt.fill_between(R, mean-stand_err, mean+stand_err, label = r"$FDM \ M: {:.2} - {:.2}$".format(max_min_m[group], max_min_m[group+1]), alpha = 0.5)
        plt.legend()
        plt.xlabel(r"$r/r_{\mathrm{ell}}^{\mathrm{max}}$")
        plt.ylabel(r"s")
        plt.title(r"Averaged s, T: {:.2}-{:.2}".format(config.T_CUT_LOW, config.T_CUT_HIGH))
        plt.legend()
        plt.savefig("{0}/dm/SHsMs.pdf".format(config.SHAPE_DEST))
        
        # T: M-splitting
        plt.figure()
        for group in range(len(sh_m_groups)):
            y = [[] for i in range(config.R_BIN)]
            for sh in idx_groups[group]:
                for rad in range(config.D_BINS):
                    closest_idx = (np.abs(R - d[sh][rad]/d_max[sh])).argmin() # Determine which point in R is closest
                    if isnan(T[sh][rad]):
                        continue
                    else:
                        y[closest_idx].append(T[sh][rad])
            mean = np.array([np.average(z) for z in y])
            stand_err = np.array([np.std(z)/np.sqrt(len(z)) for z in y])
            
            plt.semilogx(R, mean)
            plt.fill_between(R, mean-stand_err, mean+stand_err, label = r"$FDM \ M: {:.2} - {:.2}$".format(max_min_m[group], max_min_m[group+1]), alpha = 0.5)
        plt.xlabel(r"$r/r_{\mathrm{ell}}^{\mathrm{max}}$")
        plt.ylabel(r"T")
        plt.title(r"Averaged T, T: {:.2}-{:.2}".format(config.T_CUT_LOW, config.T_CUT_HIGH))
        plt.legend()
        plt.savefig("{0}/dm/SHsTMs.pdf".format(config.SHAPE_DEST))
        
        # T counting
        plt.figure()
        binwidth = 0.05
        t[t == 0.] = np.nan
        n, bins, patches = plt.hist(x=t, bins = np.arange(np.nanmin(t), np.nanmax(t) + binwidth, binwidth), alpha=0.7, density=True, label = "FDM SHs")
        plt.axvline(1/3, label="oblate-triaxial transition", color = "g")
        plt.axvline(2/3, label="triaxial-prolate transition", color = "r")
        plt.xlabel(r"T")
        plt.ylabel('Normalized Bin Count')
        plt.grid(axis='y', alpha=0.75)
        #plt.yscale('log', nonposy='clip') # strengthens visually the small alignment 
        plt.title(r"Triaxiality Histogram")
        plt.legend()
        plt.savefig("{0}/dm/{1}CSHTCount.pdf".format(config.SHAPE_DEST, config.HALO_REGION))