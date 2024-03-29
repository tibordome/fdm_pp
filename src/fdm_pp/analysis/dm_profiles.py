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
from function_utilities import readDataFDM, assembleDataFDM, getRhoProfile, getRhoProfileMs, getRhoProfileOneObj
from splitting import M_split
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
import config
from print_msg import print_status

    
def dm_profiles(start_time):
    """
    Create a series of plots to analyze Sh profiles"""
    
    print_status(rank,start_time,'Starting dm_profiles() with snap {0}'.format(config.SNAP))
    assert config.T_CUT_LOW == 0.0 and config.T_CUT_HIGH == 1.0, \
        "Do not call dm_profiles with a weird T-window"
    
    if rank == 0:
        
        # Reading
        a_com_cat_fdm, h_cat_fdm, d_fdm, q_fdm, s_fdm, major_fdm_full, rdelta_fdm, sh_masses_fdm, rho_profs = readDataFDM(get_skeleton = False, get_profiles = True)
        eps_9_fdm = np.loadtxt('{0}/eps_9_fdm_dm_{1}.txt'.format(config.CAT_DEST, config.SNAP))
        print_status(rank, start_time, "The number of FDM shs considered is {0}".format(d_fdm.shape[0]))
        
        # Assembly
        sh_masses_fdm, sh_com_arr_fdm, idx_fdm, major_fdm, t_fdm = assembleDataFDM(a_com_cat_fdm, h_cat_fdm, d_fdm, q_fdm, s_fdm, major_fdm_full, sh_masses_fdm)
        
        # Mass splitting
        max_min_m_fdm, sh_m_groups_fdm, sh_com_groups_fdm, major_groups_fdm, idx_groups_fdm = M_split(config.MASS_UNIT*sh_masses_fdm, sh_com_arr_fdm, "halo", major_fdm)
        
        # Elliptical radii
        R = np.logspace(-6,config.R_LOGEND,config.R_BIN*3+1) # Note that np.logspace(-6,1,config.R_BIN*3+1) does not mean the plots below will start at 10^-6; they are determined by config.DEL_X
        d_lin = np.linspace(1, config.N, config.N)*config.DEL_X
                
        if config.HALO_REGION == "Inner":
            # Average Profiles
            plt.figure()
            mean_median, err_low, err_high = getRhoProfile(R, d_lin, rdelta_fdm, rho_profs)
            plt.loglog(R, mean_median)
            plt.fill_between(R, mean_median-err_low, mean_median+err_high, label="FDM", edgecolor='g', alpha = 0.5)
            plt.xlabel(r"$r/R_{200}$")
            plt.ylabel(r"$\rho$ [$10^{10} M_{{\odot}}$ / Mpc/$h^2$]")
            eps_9_fdm_av = np.average(eps_9_fdm/rdelta_fdm)
            plt.axvline(x=eps_9_fdm_av, color='r', linestyle='--', label=r'$9\epsilon$ FDM')
            plt.legend(); plt.legend(fontsize="small")
            plt.savefig("{0}/dm/ShRhoprof_{1}.pdf".format(config.SHAPE_DEST, config.SNAP), bbox_inches="tight")
            
            # Average Profiles: M-splitting
            plt.figure()
            for group in range(len(sh_m_groups_fdm)):
                mean_median, err_low, err_high = getRhoProfileMs(R, d_lin, rdelta_fdm, idx_groups_fdm, group, rho_profs)
                if len(idx_groups_fdm[group]) != 0:
                    plt.loglog(R, mean_median)
                    plt.fill_between(R, mean_median-err_low, mean_median+err_high, label = r"$FDM \ M: {0} - {1} \ M_{{\odot}}$".format(eTo10("{:.2E}".format(max_min_m_fdm[group])), eTo10("{:.2E}".format(max_min_m_fdm[group+1]))), alpha = 0.5)
            plt.legend(); plt.legend(fontsize="small")
            plt.xlabel(r"$r/R_{200}$")
            plt.ylabel(r"$\rho$ [$10^{10} M_{{\odot}}$ / Mpc/$h^2$]")
            plt.axvline(x=eps_9_fdm_av, color='r', linestyle='--', label=r'$9\epsilon$ FDM')
            plt.legend(); plt.legend(fontsize="small")
            plt.savefig("{0}/dm/ShRhoprofMs_{1}.pdf".format(config.SHAPE_DEST, config.SNAP), bbox_inches="tight")
            
            # Individual Profiles
            # Elliptical radii
            for sh in range(rho_profs.shape[0]):
                plt.figure()
                plt.loglog(R, getRhoProfileOneObj(R, d_lin, rdelta_fdm[sh], rho_profs[sh]), label="SH {0}".format(sh))
                plt.xlabel(r"$r/R_{200}$")
                plt.ylabel(r"$\rho$ [$10^{10} M_{{\odot}}$ / Mpc/$h^2$]")
                eps_9_fdm_indiv = eps_9_fdm/rdelta_fdm[sh]
                plt.axvline(x=eps_9_fdm_indiv, color='r', linestyle='--', label=r'$9\epsilon$ FDM')
                plt.legend(); plt.legend(fontsize="small")
                plt.savefig("{0}/dm/Sh{1}Rhoprof_{2}.pdf".format(config.SHAPE_DEST, sh, config.SNAP), bbox_inches="tight") 