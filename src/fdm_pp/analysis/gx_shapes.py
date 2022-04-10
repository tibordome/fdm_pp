#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 18:11:27 2021

@author: tibor
"""

import numpy as np
import matplotlib.pyplot as plt
from scientific_notation import eTo10
from splitting import M_split
from print_msg import print_status
from mpi4py import MPI
import time
from function_utilities import readDataGx, assembleDataGx, getProfile, getProfileMs, getProfileOneObj
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
import config
        
def gx_shapes():
    """
    Create a series of plots to analyze gx shapes"""
    
    start_time = time.time()
    print_status(rank,start_time,'Starting gx_shapes() with snap {0}'.format(config.SNAP))
        
    assert config.T_CUT_LOW == 0.0 and config.T_CUT_HIGH == 1.0, \
        "Do not call gx_shapes with a weird T-window"
    if rank == 0:
        
        # Reading
        a_com_cat_fdm, gx_cat_fdm, d_fdm, q_fdm, s_fdm, sh_masses_fdm = readDataGx()
        eps_9_fdm = np.loadtxt('{0}/eps_9_fdm_gx_{1}.txt'.format(config.CAT_DEST, config.SNAP))
        print_status(rank, start_time, "The number of FDM gxs considered is {0}".format(len(a_com_cat_fdm)))
        
        # Assembly
        sh_masses_fdm, gx_com_arr_fdm, major_fdm, t_fdm = assembleDataGx(gx_cat_fdm, a_com_cat_fdm, q_fdm, s_fdm, sh_masses_fdm)
        
        # M-splitting
        max_min_m_fdm, gx_m_groups_fdm, gx_com_groups_fdm, major_groups_fdm, idx_groups_fdm = M_split(config.MASS_UNIT*sh_masses_fdm, gx_com_arr_fdm, "gx", major_fdm)
        
        # Elliptical radii
        R = np.logspace(config.R_LOGSTART,config.R_LOGEND,config.R_BIN+1)
        
        # Averaged profiles
        # Q
        plt.figure()
        mean_median, err_low, err_high = getProfile(R, d_fdm, q_fdm)
        plt.semilogx(R, mean_median)
        plt.fill_between(R, mean_median-err_low, mean_median+err_high, label="FDM", edgecolor='g', alpha = 0.5)
        plt.xlabel(r"$r/R_{200}$")
        plt.ylabel(r"q")
        if d_fdm.ndim == 2:
            eps_9_fdm_av = np.average(eps_9_fdm/d_fdm[:,-int(config.D_LOGEND/((config.D_LOGEND-config.D_LOGSTART)/config.D_BINS))-1])
        else:
            eps_9_fdm_av = np.nan # We are dealing with empty arrays, i.e. no gxs considered
        plt.axvline(x=eps_9_fdm_av, color='r', linestyle='--', label=r'$9\epsilon$ FDM')
        plt.legend(); plt.legend(fontsize="small")
        plt.ylim(0.0, 1.0)
        plt.savefig("{0}/gxs/Gxq_{1}.pdf".format(config.SHAPE_DEST, config.SNAP), bbox_inches="tight")   
        
        # S
        plt.figure()
        mean_median, err_low, err_high = getProfile(R, d_fdm, s_fdm)
        plt.semilogx(R, mean_median)
        plt.fill_between(R, mean_median-err_low, mean_median+err_high, label="FDM", edgecolor='g', alpha = 0.5)
        plt.xlabel(r"$r/R_{200}$")
        plt.ylabel(r"s")
        plt.axvline(x=eps_9_fdm_av, color='r', linestyle='--', label=r'$9\epsilon$ FDM')
        plt.legend(); plt.legend(fontsize="small")
        plt.ylim(0.0, 1.0)
        plt.savefig("{0}/gxs/Gxs_{1}.pdf".format(config.SHAPE_DEST, config.SNAP), bbox_inches="tight")   
        
        # T
        plt.figure()
        if q_fdm.ndim == 2:
            T_fdm = np.zeros((q_fdm.shape[0], q_fdm.shape[1]))
            for gx in range(q_fdm.shape[0]):
                T_fdm[gx] = (1-q_fdm[gx]**2)/(1-s_fdm[gx]**2) # Triaxiality
        else:
            T_fdm = np.empty(0)
        mean_median, err_low, err_high = getProfile(R, d_fdm, T_fdm)
        plt.semilogx(R, mean_median)
        plt.fill_between(R, mean_median-err_low, mean_median+err_high, label="FDM", edgecolor='g', alpha = 0.5)
        plt.xlabel(r"$r/R_{200}$")
        plt.ylabel(r"T")
        plt.axvline(x=eps_9_fdm_av, color='r', linestyle='--', label=r'$9\epsilon$ FDM')
        plt.legend(); plt.legend(fontsize="small")
        plt.ylim(0.0, 1.0)
        plt.axhline(2/3, label=r"$T$ > 2/3: prolate", linestyle='--', color = "y")
        plt.savefig("{0}/gxs/GxT_{1}.pdf".format(config.SHAPE_DEST, config.SNAP), bbox_inches="tight")
        
        
        # Q: M-splitting
        plt.figure()
        for group in range(len(gx_m_groups_fdm)):
            mean_median, err_low, err_high = getProfileMs(R, d_fdm, idx_groups_fdm, group, q_fdm)
            if len(idx_groups_fdm[group]) != 0:
                plt.semilogx(R, mean_median)
                plt.fill_between(R, mean_median-err_low, mean_median+err_high, label = r"$FDM \ M: {0} - {1} \ M_{{\odot}}$".format(eTo10("{:.2E}".format(max_min_m_fdm[group])), eTo10("{:.2E}".format(max_min_m_fdm[group+1]))), alpha = 0.5)
        plt.legend(); plt.legend(fontsize="small")
        plt.xlabel(r"$r/R_{200}$")
        plt.ylabel(r"q")
        plt.axvline(x=eps_9_fdm_av, color='r', linestyle='--', label=r'$9\epsilon$ FDM')
        plt.legend(); plt.legend(fontsize="small")
        plt.ylim(0.0, 1.0)
        plt.savefig("{0}/gxs/GxqMs_{1}.pdf".format(config.SHAPE_DEST, config.SNAP), bbox_inches="tight")   
        
        # S: M-splitting
        plt.figure()
        for group in range(len(gx_m_groups_fdm)):
            mean_median, err_low, err_high = getProfileMs(R, d_fdm, idx_groups_fdm, group, s_fdm)
            if len(idx_groups_fdm[group]) != 0:
                plt.semilogx(R, mean_median)
                plt.fill_between(R, mean_median-err_low, mean_median+err_high, label = r"$FDM \ M: {0} - {1} \ M_{{\odot}}$".format(eTo10("{:.2E}".format(max_min_m_fdm[group])), eTo10("{:.2E}".format(max_min_m_fdm[group+1]))), alpha = 0.5)
        plt.legend(); plt.legend(fontsize="small")
        plt.xlabel(r"$r/R_{200}$")
        plt.ylabel(r"s")
        plt.axvline(x=eps_9_fdm_av, color='r', linestyle='--', label=r'$9\epsilon$ FDM')
        plt.legend(); plt.legend(fontsize="small")
        plt.ylim(0.0, 1.0)
        plt.savefig("{0}/gxs/GxsMs_{1}.pdf".format(config.SHAPE_DEST, config.SNAP), bbox_inches="tight")   
        
        # T: M-splitting
        plt.figure()
        for group in range(len(gx_m_groups_fdm)):
            mean_median, err_low, err_high = getProfileMs(R, d_fdm, idx_groups_fdm, group, T_fdm)
            if len(idx_groups_fdm[group]) != 0:
                plt.semilogx(R, mean_median)
                plt.fill_between(R, mean_median-err_low, mean_median+err_high, label = r"$FDM \ M: {0} - {1} \ M_{{\odot}}$".format(eTo10("{:.2E}".format(max_min_m_fdm[group])), eTo10("{:.2E}".format(max_min_m_fdm[group+1]))), alpha = 0.5)
        plt.legend(); plt.legend(fontsize="small")
        plt.xlabel(r"$r/R_{200}$")
        plt.ylabel(r"T")
        plt.axvline(x=eps_9_fdm_av, color='r', linestyle='--', label=r'$9\epsilon$ FDM')
        plt.legend(); plt.legend(fontsize="small")
        plt.ylim(0.0, 1.0)
        plt.axhline(2/3, label=r"$T$ > 2/3: prolate", linestyle='--', color = "y")
        plt.savefig("{0}/gxs/GxTMs_{1}.pdf".format(config.SHAPE_DEST, config.SNAP), bbox_inches="tight")   
        
        # T counting
        plt.figure()
        t_fdm[t_fdm == 0.] = np.nan
        n, bins, patches = plt.hist(x=t_fdm, bins = np.linspace(0, 1, config.HIST_NB_BINS), alpha=0.7, density=True, label = "FDM Gxs")
        plt.axvline(1/3, label="oblate-triaxial transition", color = "g")
        plt.axvline(2/3, label="triaxial-prolate transition", color = "r")
        plt.xlabel(r"T")
        plt.ylabel('Normalized Bin Count')
        plt.grid(axis='y', alpha=0.75)
        #plt.yscale('log', nonposy='clip') # strengthens visually the small alignment 
        plt.legend(); plt.legend(fontsize="small")
        plt.xlim(0.0, 1.0)
        plt.savefig("{0}/gxs/GxTCount_{1}.pdf".format(config.SHAPE_DEST, config.SNAP), bbox_inches="tight")
        
        t_fdm = t_fdm[np.logical_not(np.isnan(t_fdm))]
        print_status(rank, start_time, "In degrees: The average T value for FDM Gxs is {0} and the standard deviation (assuming T is Gaussian distributed) is {1}".format(round(np.average(t_fdm),2), round(np.std(t_fdm),2)))
        
        # Individual profiles
        print("T_fdm is", T_fdm)
        print("T_fdm[0] is", q_fdm[0], s_fdm[0], T_fdm[0])
        
        for gx in range(q_fdm.shape[0]):
            plt.figure()
            plt.semilogx(R, getProfileOneObj(R, d_fdm[gx], q_fdm[gx]), label="Gx {0}".format(gx))
            plt.xlabel(r"$r/R_{200}$")
            plt.ylabel(r"q")
            if d_fdm.ndim == 2:
                eps_9_fdm_indiv = eps_9_fdm/d_fdm[gx,-int(config.D_LOGEND/((config.D_LOGEND-config.D_LOGSTART)/config.D_BINS))-1]
            else:
                eps_9_fdm_indiv = np.nan # We are dealing with empty arrays, i.e. no gxs considered
            plt.axvline(x=eps_9_fdm_indiv, color='r', linestyle='--', label=r'$9\epsilon$ FDM')
            plt.legend(); plt.legend(fontsize="small")
            plt.ylim(0.0, 1.0)
            plt.savefig("{0}/gxs/Gx{1}q_{2}.pdf".format(config.SHAPE_DEST, gx, config.SNAP), bbox_inches="tight")
            
            plt.figure()
            plt.semilogx(R, getProfileOneObj(R, d_fdm[gx], s_fdm[gx]), label="Gx {0}".format(gx))
            plt.xlabel(r"$r/R_{200}$")
            plt.ylabel(r"s")
            if d_fdm.ndim == 2:
                eps_9_fdm_indiv = eps_9_fdm/d_fdm[gx,-int(config.D_LOGEND/((config.D_LOGEND-config.D_LOGSTART)/config.D_BINS))-1]
            else:
                eps_9_fdm_indiv = np.nan # We are dealing with empty arrays, i.e. no gxs considered
            plt.axvline(x=eps_9_fdm_indiv, color='r', linestyle='--', label=r'$9\epsilon$ FDM')
            plt.legend(); plt.legend(fontsize="small")
            plt.ylim(0.0, 1.0)
            plt.savefig("{0}/gxs/Gx{1}s_{2}.pdf".format(config.SHAPE_DEST, gx, config.SNAP), bbox_inches="tight")
            
            plt.figure()
            plt.semilogx(getProfileOneObj(R, d_fdm[gx], T_fdm[gx]), label="Gx {0}".format(gx))
            plt.xlabel(r"$r/R_{200}$")
            plt.ylabel(r"T")
            if d_fdm.ndim == 2:
                eps_9_fdm_indiv = eps_9_fdm/d_fdm[gx,-int(config.D_LOGEND/((config.D_LOGEND-config.D_LOGSTART)/config.D_BINS))-1]
            else:
                eps_9_fdm_indiv = np.nan # We are dealing with empty arrays, i.e. no gxs considered
            plt.axvline(x=eps_9_fdm_indiv, color='r', linestyle='--', label=r'$9\epsilon$ FDM')
            plt.legend(); plt.legend(fontsize="small")
            plt.ylim(0.0, 1.0)
            plt.savefig("{0}/gxs/Gx{1}T_{2}.pdf".format(config.SHAPE_DEST, gx, config.SNAP), bbox_inches="tight")