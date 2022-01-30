#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 09:54:40 2021

@author: tibor
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 13})
from scientific_notation import eTo10
from get_hdf5 import getHDF5DMData
from function_utilities import readDataFDM, assembleDataFDM, getProfile, getProfileMs, getEpsilon
from splitting import M_split
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
import config
from config import makeGlobalSNAP
from print_msg import print_status

    
def dm_shapes(start_time):
    """
    Create a series of plots to analyze Sh shapes"""
    
    print_status(rank,start_time,'Starting dm_shapes() with snap {0}'.format(config.SNAP))
    assert config.T_CUT_LOW == 0.0 and config.T_CUT_HIGH == 1.0, \
        "Do not call dm_shapes with a weird T-window"
    
    if rank == 0:
        
        # Reading
        a_com_cat_fdm, h_cat_fdm, d_fdm, q_fdm, s_fdm, major_fdm_full, r200_fdm, sh_masses_fdm = readDataFDM(get_skeleton = False)
        eps_9_fdm = np.loadtxt('{0}/eps_9_fdm_dm_{1}.txt'.format(config.CAT_DEST, config.SNAP))
        print_status(rank, start_time, "The number of FDM shs considered is {0}".format(len(a_com_cat_fdm)))
        
        # Assembly
        sh_masses_fdm, sh_com_arr_fdm, idx_fdm, major_fdm, t_fdm = assembleDataFDM(a_com_cat_fdm, h_cat_fdm, d_fdm, q_fdm, s_fdm, major_fdm_full, sh_masses_fdm)
        
        # Mass splitting
        max_min_m_fdm, sh_m_groups_fdm, sh_com_groups_fdm, major_groups_fdm, idx_groups_fdm = M_split(config.MASS_UNIT*sh_masses_fdm, sh_com_arr_fdm, "halo", major_fdm)
        
        # Maximal elliptical radii
        R = np.logspace(config.R_LOGSTART,config.R_LOGEND,config.R_BIN+1)
        
        # Averaged profiles
        if config.HALO_REGION == "Inner":
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
                eps_9_fdm_av = np.nan # We are dealing with empty arrays, i.e. no shs considered
            plt.axvline(x=eps_9_fdm_av, color='r', linestyle='--', label=r'$9\epsilon$ FDM')
            plt.legend(); plt.legend(fontsize="small")
            plt.ylim(0.0, 1.0)
            plt.savefig("{0}/dm/Shq_{1}.pdf".format(config.SHAPE_DEST, config.SNAP), bbox_inches="tight")
            
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
            plt.savefig("{0}/dm/Shs_{1}.pdf".format(config.SHAPE_DEST, config.SNAP), bbox_inches="tight")
            
            # T
            plt.figure()
            if q_fdm.ndim == 2:
                T_fdm = np.zeros((q_fdm.shape[0], q_fdm.shape[1]))
                for sh in range(q_fdm.shape[0]):
                    T_fdm[sh] = (1-q_fdm[sh]**2)/(1-s_fdm[sh]**2) # Triaxiality
            else:
                T_fdm = np.empty(0)
            mean_median, err_low, err_high = getProfile(R, d_fdm, T_fdm)
            plt.semilogx(R, mean_median)
            plt.fill_between(R, mean_median-err_low, mean_median+err_high, label="FDM", edgecolor='g', alpha = 0.5)
            plt.xlabel(r"$r/R_{200}$")
            plt.ylabel(r"T")
            plt.axhline(2/3, label=r"$T$ > 2/3: prolate", linestyle='--', color = "y")
            plt.axvline(x=eps_9_fdm_av, color='r', linestyle='--', label=r'$9\epsilon$ FDM')
            plt.legend(); plt.legend(fontsize="small")
            plt.ylim(0.0, 1.0)
            plt.savefig("{0}/dm/ShT_{1}.pdf".format(config.SHAPE_DEST, config.SNAP), bbox_inches="tight")
            
            # Q: M-splitting
            plt.figure()
            for group in range(len(sh_m_groups_fdm)):
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
            plt.savefig("{0}/dm/ShqMs_{1}.pdf".format(config.SHAPE_DEST, config.SNAP), bbox_inches="tight")
            
            # S: M-splitting
            plt.figure()
            for group in range(len(sh_m_groups_fdm)):
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
            plt.savefig("{0}/dm/ShsMs_{1}.pdf".format(config.SHAPE_DEST, config.SNAP), bbox_inches="tight")
            
            # T: M-splitting
            plt.figure()
            for group in range(len(sh_m_groups_fdm)):
                mean_median, err_low, err_high = getProfileMs(R, d_fdm, idx_groups_fdm, group, T_fdm)
                if len(idx_groups_fdm[group]) != 0:
                    plt.semilogx(R, mean_median)
                    plt.fill_between(R, mean_median-err_low, mean_median+err_high, label = r"$FDM \ M: {0} - {1} \ M_{{\odot}}$".format(eTo10("{:.2E}".format(max_min_m_fdm[group])), eTo10("{:.2E}".format(max_min_m_fdm[group+1]))), alpha = 0.5)
            plt.xlabel(r"$r/R_{200}$")
            plt.ylabel(r"T")
            plt.axhline(2/3, label=r"$T$ > 2/3: prolate", linestyle='--', color = "y")
            plt.axvline(x=eps_9_fdm_av, color='r', linestyle='--', label=r'$9\epsilon$ FDM')
            plt.legend(); plt.legend(fontsize="small")
            plt.ylim(0.0, 1.0)
            plt.savefig("{0}/dm/ShsTMs_{1}.pdf".format(config.SHAPE_DEST, config.SNAP), bbox_inches="tight")
        
        # T counting
        plt.figure()
        t_fdm[t_fdm == 0.] = np.nan
        n, bins, patches = plt.hist(x=t_fdm, bins = np.linspace(0, 1, config.HIST_NB_BINS), alpha=0.7, density=True, label = "FDM Shs")
        plt.axvline(1/3, label="oblate-triaxial transition", color = "g")
        plt.axvline(2/3, label="triaxial-prolate transition", color = "r")
        plt.xlabel(r"T")
        plt.ylabel('Normalized Bin Count')
        plt.grid(axis='y', alpha=0.75)
        plt.legend(); plt.legend(fontsize="small")
        plt.xlim(0.0, 1.0)
        plt.savefig("{0}/dm/{1}ShTCount_{2}.pdf".format(config.SHAPE_DEST, config.HALO_REGION, config.SNAP), bbox_inches="tight")
                
        t_fdm = t_fdm[np.logical_not(np.isnan(t_fdm))]
        print_status(rank, start_time, "{0}. In degrees: The average T value for FDM Shs is {1} and the standard deviation (assuming T is Gaussian distributed) is {2}".format(config.HALO_REGION, round(np.average(t_fdm),2), round(np.std(t_fdm),2)))
        
        # Individual profiles
        for sh in range(q_fdm.shape[0]):
            plt.figure()
            plt.semilogx(R, q_fdm[sh], label="SH {0}".format(sh))
            plt.xlabel(r"$r/R_{200}$")
            plt.ylabel(r"q")
            if d_fdm.ndim == 2:
                eps_9_fdm_indiv = eps_9_fdm/d_fdm[sh,-int(config.D_LOGEND/((config.D_LOGEND-config.D_LOGSTART)/config.D_BINS))-1]
            else:
                eps_9_fdm_indiv = np.nan # We are dealing with empty arrays, i.e. no shs considered
            plt.axvline(x=eps_9_fdm_indiv, color='r', linestyle='--', label=r'$9\epsilon$ FDM')
            plt.legend(); plt.legend(fontsize="small")
            plt.ylim(0.0, 1.0)
            plt.savefig("{0}/dm/Sh{1}q_{2}.pdf".format(config.SHAPE_DEST, sh, config.SNAP), bbox_inches="tight")
            
            plt.figure()
            plt.semilogx(R, s_fdm[sh], label="SH {0}".format(sh))
            plt.xlabel(r"$r/R_{200}$")
            plt.ylabel(r"q")
            if d_fdm.ndim == 2:
                eps_9_fdm_indiv = eps_9_fdm/d_fdm[sh,-int(config.D_LOGEND/((config.D_LOGEND-config.D_LOGSTART)/config.D_BINS))-1]
            else:
                eps_9_fdm_indiv = np.nan # We are dealing with empty arrays, i.e. no shs considered
            plt.axvline(x=eps_9_fdm_indiv, color='r', linestyle='--', label=r'$9\epsilon$ FDM')
            plt.legend(); plt.legend(fontsize="small")
            plt.ylim(0.0, 1.0)
            plt.savefig("{0}/dm/Sh{1}s_{2}.pdf".format(config.SHAPE_DEST, sh, config.SNAP), bbox_inches="tight")
            
            plt.figure()
            plt.semilogx(R, t_fdm[sh], label="SH {0}".format(sh))
            plt.xlabel(r"$r/R_{200}$")
            plt.ylabel(r"q")
            if d_fdm.ndim == 2:
                eps_9_fdm_indiv = eps_9_fdm/d_fdm[sh,-int(config.D_LOGEND/((config.D_LOGEND-config.D_LOGSTART)/config.D_BINS))-1]
            else:
                eps_9_fdm_indiv = np.nan # We are dealing with empty arrays, i.e. no shs considered
            plt.axvline(x=eps_9_fdm_indiv, color='r', linestyle='--', label=r'$9\epsilon$ FDM')
            plt.legend(); plt.legend(fontsize="small")
            plt.ylim(0.0, 1.0)
            plt.savefig("{0}/dm/Sh{1}q_{2}.pdf".format(config.SHAPE_DEST, sh, config.SNAP), bbox_inches="tight")
            
    if config.HALO_REGION == "Full":
        # \epsilon histogram
        makeGlobalSNAP(config.SNAP, start_time)
        dm_xyz, dm_masses, dm_smoothing, dm_velxyz = getHDF5DMData()
        if rank == 0:
            eps_fdm = getEpsilon(h_cat_fdm, dm_xyz, dm_masses)
            n, bins, patches = plt.hist(x=abs(eps_fdm), bins = np.linspace(0, 1, config.HIST_NB_BINS), alpha=0.7, density=True, label = "FDM Shs")
            plt.xlabel(r"$\epsilon$")
            plt.ylabel('Normalized Bin Count')
            plt.grid(axis='y', alpha=0.75)
            plt.legend(); plt.legend(fontsize="small")
            plt.xlim(0.0, 1.0)
            plt.savefig("{0}/dm/{1}ShEpsCount_{2}.pdf".format(config.SHAPE_DEST, config.HALO_REGION, config.SNAP), bbox_inches="tight")