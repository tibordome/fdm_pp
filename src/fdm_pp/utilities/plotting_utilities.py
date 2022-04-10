#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 13:27:49 2021

@author: tibor
"""
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import config
import numpy as np
import accum
import pandas as pd

def plotThreshold(vec, data, category):
    plt.figure()
    plt.semilogx(vec, data/np.max(data)) # Arbitary units: so normalize to 1
    plt.xlabel(r'{0} Signature $S_{{{1}}}$'.format(category.capitalize(), category[0]))
    if category == "cluster":
        plt.ylabel(r'fraction of valid clusters') 
    else:
        plt.ylabel(r'$\Delta M^2_{{{}}}$ (arbitrary units)'.format(category[0]))
    plt.savefig("{0}/{1}ThresholdS{2}Cut_{3}.pdf".format(config.NEXUS_DEST, config.DM_TYPE.upper(), category[0], config.SNAP))

def plotNNProjections(x_vec, y_vec, z_vec, mass_array):
    """Plots the NN projections of the point distribution
    Arguments:
    -------------
    x_vec, y_vec, z_vec: coordinates of the points in cMpc/h
    Returns: 
    -------------
    3 NN projection plots"""
    
    # NN projections
    rho_NNz = np.zeros((config.N,config.N))
    stack = np.hstack((np.reshape(np.rint(x_vec*config.N/config.L_BOX-0.5), (len(x_vec),1)),np.reshape(np.rint(y_vec*config.N/config.L_BOX-0.5), (len(y_vec),1))))
    # Let's make sure that we do not drop the 2 points with x or y = L_BOX..
    stack[stack == config.N] = config.N-1
    accummap = pd.DataFrame(stack)
    a = pd.Series(mass_array/((config.L_BOX/config.N)**3))
    rho_NNz += accum.accumarray(accummap, a, size=[config.N,config.N])/config.N
    rho_NNz = np.array(rho_NNz)
    
    rho_NNy = np.zeros((config.N,config.N))
    stack = np.hstack((np.reshape(np.rint(x_vec*config.N/config.L_BOX-0.5), (len(x_vec),1)),np.reshape(np.rint(z_vec*config.N/config.L_BOX-0.5), (len(z_vec),1))))
    stack[stack == config.N] = config.N-1
    accummap = pd.DataFrame(stack)
    rho_NNy += accum.accumarray(accummap, a, size=[config.N,config.N])/config.N
    rho_NNy = np.array(rho_NNy)

    rho_NNx = np.zeros((config.N,config.N))
    stack = np.hstack((np.reshape(np.rint(y_vec*config.N/config.L_BOX-0.5), (len(y_vec),1)),np.reshape(np.rint(z_vec*config.N/config.L_BOX-0.5), (len(z_vec),1))))
    stack[stack == config.N] = config.N-1
    accummap = pd.DataFrame(stack)
    rho_NNx += accum.accumarray(accummap, a, size=[config.N,config.N])/config.N
    rho_NNx = np.array(rho_NNx)

    plt.figure()
    second_smallest = np.unique(rho_NNz)[1]
    rho_NNz[rho_NNz < second_smallest] = second_smallest
    plt.imshow(rho_NNz,interpolation='None',origin='upper', extent=[0, config.L_BOX, config.L_BOX, 0], cmap="hot", norm=colors.LogNorm(vmin=second_smallest, vmax=np.max(rho_NNz)))
    plt.gca().xaxis.set_major_locator(MultipleLocator(config.L_BOX/4))
    plt.gca().xaxis.set_minor_locator(MultipleLocator(config.L_BOX/20))
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f')) # float
    plt.gca().yaxis.set_major_locator(MultipleLocator(config.L_BOX/4))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(config.L_BOX/20))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f')) # float
    plt.xlabel(r"y (cMpc/h)", fontweight='bold')
    plt.ylabel(r"x (cMpc/h)", fontweight='bold')
    plt.colorbar()
    plt.title(r'z-Projected NN-Density')
    plt.savefig("{0}/{1}zProjRhoNN_{2}.pdf".format(config.NEXUS_DEST, config.DM_TYPE.upper(), config.SNAP))
    
    plt.figure()
    second_smallest = np.unique(rho_NNy)[1]
    rho_NNy[rho_NNy < second_smallest] = second_smallest
    plt.imshow(rho_NNy,interpolation='None',origin='upper', extent=[0, config.L_BOX, config.L_BOX, 0], cmap="hot", norm=colors.LogNorm(vmin=second_smallest, vmax=np.max(rho_NNy)))
    plt.gca().xaxis.set_major_locator(MultipleLocator(config.L_BOX/4))
    plt.gca().xaxis.set_minor_locator(MultipleLocator(config.L_BOX/20))
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f')) # float
    plt.gca().yaxis.set_major_locator(MultipleLocator(config.L_BOX/4))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(config.L_BOX/20))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f')) # float
    plt.xlabel(r"z (cMpc/h)", fontweight='bold')
    plt.ylabel(r"x (cMpc/h)", fontweight='bold')
    plt.colorbar()
    plt.title(r'y-Projected NN-Density')
    plt.savefig("{0}/{1}yProjRhoNN_{2}.pdf".format(config.NEXUS_DEST, config.DM_TYPE.upper(), config.SNAP))
    
    plt.figure()
    second_smallest = np.unique(rho_NNx)[1]
    rho_NNx[rho_NNx < second_smallest] = second_smallest
    plt.imshow(rho_NNx,interpolation='None',origin='upper', extent=[0, config.L_BOX, config.L_BOX, 0], cmap="hot", norm=colors.LogNorm(vmin=second_smallest, vmax=np.max(rho_NNx)))
    plt.gca().xaxis.set_major_locator(MultipleLocator(config.L_BOX/4))
    plt.gca().xaxis.set_minor_locator(MultipleLocator(config.L_BOX/20))
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f')) # float
    plt.gca().yaxis.set_major_locator(MultipleLocator(config.L_BOX/4))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(config.L_BOX/20))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f')) # float
    plt.xlabel(r"z (cMpc/h)", fontweight='bold')
    plt.ylabel(r"y (cMpc/h)", fontweight='bold')
    plt.colorbar()
    plt.title(r'x-Projected NN-Density')
    plt.savefig("{0}/{1}xProjRhoNN_{2}.pdf".format(config.NEXUS_DEST, config.DM_TYPE.upper(), config.SNAP))
    
def plotGridProjections(grid):
    """Plots the grid projections
    Arguments:
    -------------
    grid: (N, N, N)-density array, either calculated via CIC or SPH or DTFE
    Returns: 
    -------------
    3 grid projection plots"""
    
    rho_proj_cic = np.zeros((config.N, config.N))
    for h in range(config.N):
        rho_proj_cic += grid[h,:,:]
    rho_proj_cic /= config.N
    plt.figure()
    second_smallest = np.unique(rho_proj_cic)[1]
    rho_proj_cic[rho_proj_cic < second_smallest] = second_smallest
    plt.imshow(rho_proj_cic,interpolation='None',origin='upper', extent=[0, config.L_BOX, config.L_BOX, 0], cmap="hot", norm=colors.LogNorm(vmin=second_smallest, vmax=np.max(rho_proj_cic)))
    plt.gca().xaxis.set_major_locator(MultipleLocator(config.L_BOX/4))
    plt.gca().xaxis.set_minor_locator(MultipleLocator(config.L_BOX/20))
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f')) # float
    plt.gca().yaxis.set_major_locator(MultipleLocator(config.L_BOX/4))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(config.L_BOX/20))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f')) # float
    plt.xlabel(r"z (cMpc/h)", fontweight='bold')
    plt.ylabel(r"y (cMpc/h)", fontweight='bold')
    plt.colorbar()
    plt.title(r'x-Projected CIC-Density')
    plt.savefig("{0}/{1}xProjRho_{2}.pdf".format(config.NEXUS_DEST, config.DM_TYPE.upper(), config.SNAP))
    
    rho_proj_cic = np.zeros((config.N, config.N))
    for h in range(config.N):
        rho_proj_cic += grid[:,h,:]
    rho_proj_cic /= config.N
    plt.figure()
    second_smallest = np.unique(rho_proj_cic)[1]
    rho_proj_cic[rho_proj_cic < second_smallest] = second_smallest
    plt.imshow(rho_proj_cic,interpolation='None',origin='upper', extent=[0, config.L_BOX, config.L_BOX, 0], cmap="hot", norm=colors.LogNorm(vmin=second_smallest, vmax=np.max(rho_proj_cic)))
    plt.gca().xaxis.set_major_locator(MultipleLocator(config.L_BOX/4))
    plt.gca().xaxis.set_minor_locator(MultipleLocator(config.L_BOX/20))
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f')) # float
    plt.gca().yaxis.set_major_locator(MultipleLocator(config.L_BOX/4))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(config.L_BOX/20))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f')) # float
    plt.xlabel(r"z (cMpc/h)", fontweight='bold')
    plt.ylabel(r"x (cMpc/h)", fontweight='bold')
    plt.colorbar()
    plt.title(r'y-Projected CIC-Density')
    plt.savefig("{0}/{1}yProjRho_{2}.pdf".format(config.NEXUS_DEST, config.DM_TYPE.upper(), config.SNAP))
    
    rho_proj_cic = np.zeros((config.N, config.N))
    for h in range(config.N):
        rho_proj_cic += grid[:,:,h]
    rho_proj_cic /= config.N
    plt.figure()
    second_smallest = np.unique(rho_proj_cic)[1]
    rho_proj_cic[rho_proj_cic < second_smallest] = second_smallest
    plt.imshow(rho_proj_cic,interpolation='None',origin='upper', extent=[0, config.L_BOX, config.L_BOX, 0], cmap="hot", norm=colors.LogNorm(vmin=second_smallest, vmax=np.max(rho_proj_cic)))
    plt.gca().xaxis.set_major_locator(MultipleLocator(config.L_BOX/4))
    plt.gca().xaxis.set_minor_locator(MultipleLocator(config.L_BOX/20))
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f')) # float
    plt.gca().yaxis.set_major_locator(MultipleLocator(config.L_BOX/4))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(config.L_BOX/20))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f')) # float
    plt.xlabel(r"y (cMpc/h)", fontweight='bold')
    plt.ylabel(r"x (cMpc/h)", fontweight='bold')
    plt.colorbar()
    plt.title(r'z-Projected CIC-Density')
    plt.savefig("{0}/{1}zProjRho_{2}.pdf".format(config.NEXUS_DEST, config.DM_TYPE.upper(), config.SNAP))
    
def plotSigProj(ov_sig, txt):
    """Plots the signature projections
    Arguments:
    -------------
    ov_cluster_sig, ..., ov_wall_sig: (N, N, N)-density arrays, signatures
    txt: 6 options, which web component and whether full signature 
    (above & below threshold) or above-threshold signature
    Returns: 
    -------------
    6 signature projection plots: 3 (x,y,z) * 2 (non-log and log)"""
    
    if txt == "csig":
        save_txt1 = "Cluster Signature"
        save_txt2 = "ClusterSig"
    elif txt == "fsig":
        save_txt1 = "Filament Signature"
        save_txt2 = "FilamentSig"
    elif txt == "wsig":
        save_txt1 = "Wall Signature"
        save_txt2 = "WallSig"
    elif txt == "c":
        save_txt1 = "Clusters"
        save_txt2 = "Clusters"
    elif txt == "f":
        save_txt1 = "Filaments"
        save_txt2 = "Filaments"
    else:
        assert txt == "w"
        save_txt1 = "Walls"
        save_txt2 = "Walls"
    # Project cluster signature
    sig_proj = np.zeros((config.N, config.N))
    for h in range(config.N):
        sig_proj += ov_sig[h,:,:]
    sig_proj /= config.N
    plt.figure()
    plt.imshow(sig_proj,interpolation='None',origin='upper', extent=[0, config.L_BOX, config.L_BOX, 0], cmap="hot")
    plt.gca().xaxis.set_major_locator(MultipleLocator(config.L_BOX/4))
    plt.gca().xaxis.set_minor_locator(MultipleLocator(config.L_BOX/20))
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f')) # float
    plt.gca().yaxis.set_major_locator(MultipleLocator(config.L_BOX/4))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(config.L_BOX/20))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f')) # float
    plt.xlabel(r"z (cMpc/h)", fontweight='bold')
    plt.ylabel(r"y (cMpc/h)", fontweight='bold')
    plt.colorbar()
    plt.title(r'x-Projected {0}'.format(save_txt1))
    plt.savefig("{0}/{1}xProj{2}_{3}.pdf".format(config.NEXUS_DEST, config.DM_TYPE.upper(), save_txt2, config.SNAP))
    
    plt.figure()
    second_smallest = np.unique(sig_proj)[1]
    sig_proj[sig_proj < second_smallest] = second_smallest
    plt.imshow(sig_proj,interpolation='None',origin='upper', extent=[0, config.L_BOX, config.L_BOX, 0], cmap="hot", norm=colors.LogNorm(vmin=second_smallest, vmax=np.max(sig_proj)))
    plt.gca().xaxis.set_major_locator(MultipleLocator(config.L_BOX/4))
    plt.gca().xaxis.set_minor_locator(MultipleLocator(config.L_BOX/20))
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f')) # float
    plt.gca().yaxis.set_major_locator(MultipleLocator(config.L_BOX/4))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(config.L_BOX/20))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f')) # float
    plt.xlabel(r"z (cMpc/h)", fontweight='bold')
    plt.ylabel(r"y (cMpc/h)", fontweight='bold')
    plt.colorbar()
    plt.title(r'x-Projected {0}'.format(save_txt1))
    plt.savefig("{0}/{1}xProj{2}_{3}Log.pdf".format(config.NEXUS_DEST, config.DM_TYPE.upper(), save_txt2, config.SNAP))
    
    sig_proj = np.zeros((config.N, config.N))
    for h in range(config.N):
        sig_proj += ov_sig[:,h,:]
    sig_proj /= config.N
    plt.figure()
    plt.imshow(sig_proj,interpolation='None',origin='upper', extent=[0, config.L_BOX, config.L_BOX, 0], cmap="hot")
    plt.gca().xaxis.set_major_locator(MultipleLocator(config.L_BOX/4))
    plt.gca().xaxis.set_minor_locator(MultipleLocator(config.L_BOX/20))
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f')) # float
    plt.gca().yaxis.set_major_locator(MultipleLocator(config.L_BOX/4))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(config.L_BOX/20))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f')) # float
    plt.xlabel(r"z (cMpc/h)", fontweight='bold')
    plt.ylabel(r"x (cMpc/h)", fontweight='bold')
    plt.colorbar()
    plt.title(r'y-Projected {0}'.format(save_txt1))
    plt.savefig("{0}/{1}yProj{2}_{3}.pdf".format(config.NEXUS_DEST, config.DM_TYPE.upper(), save_txt2, config.SNAP))
    
    plt.figure()
    second_smallest = np.unique(sig_proj)[1]
    sig_proj[sig_proj < second_smallest] = second_smallest
    plt.imshow(sig_proj,interpolation='None',origin='upper', extent=[0, config.L_BOX, config.L_BOX, 0], cmap="hot", norm=colors.LogNorm(vmin=second_smallest, vmax=np.max(sig_proj)))
    plt.gca().xaxis.set_major_locator(MultipleLocator(config.L_BOX/4))
    plt.gca().xaxis.set_minor_locator(MultipleLocator(config.L_BOX/20))
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f')) # float
    plt.gca().yaxis.set_major_locator(MultipleLocator(config.L_BOX/4))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(config.L_BOX/20))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f')) # float
    plt.xlabel(r"z (cMpc/h)", fontweight='bold')
    plt.ylabel(r"x (cMpc/h)", fontweight='bold')
    plt.colorbar()
    plt.title(r'y-Projected {0}'.format(save_txt1))
    plt.savefig("{0}/{1}yProj{2}_{3}Log.pdf".format(config.NEXUS_DEST, config.DM_TYPE.upper(), save_txt2, config.SNAP))
    
    sig_proj = np.zeros((config.N, config.N))
    for h in range(config.N):
        sig_proj += ov_sig[:,:,h]
    sig_proj /= config.N
    plt.figure()
    plt.imshow(sig_proj,interpolation='None',origin='upper', extent=[0, config.L_BOX, config.L_BOX, 0], cmap="hot")
    plt.gca().xaxis.set_major_locator(MultipleLocator(config.L_BOX/4))
    plt.gca().xaxis.set_minor_locator(MultipleLocator(config.L_BOX/20))
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f')) # float
    plt.gca().yaxis.set_major_locator(MultipleLocator(config.L_BOX/4))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(config.L_BOX/20))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f')) # float
    plt.xlabel(r"y (cMpc/h)", fontweight='bold')
    plt.ylabel(r"x (cMpc/h)", fontweight='bold')
    plt.colorbar()
    plt.title(r'z-Projected {0}'.format(save_txt1))
    plt.savefig("{0}/{1}zProj{2}_{3}.pdf".format(config.NEXUS_DEST, config.DM_TYPE.upper(), save_txt2, config.SNAP))
    
    plt.figure()
    second_smallest = np.unique(sig_proj)[1]
    sig_proj[sig_proj < second_smallest] = second_smallest
    plt.imshow(sig_proj,interpolation='None',origin='upper', extent=[0, config.L_BOX, config.L_BOX, 0], cmap="hot", norm=colors.LogNorm(vmin=second_smallest, vmax=np.max(sig_proj)))
    plt.gca().xaxis.set_major_locator(MultipleLocator(config.L_BOX/4))
    plt.gca().xaxis.set_minor_locator(MultipleLocator(config.L_BOX/20))
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f')) # float
    plt.gca().yaxis.set_major_locator(MultipleLocator(config.L_BOX/4))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(config.L_BOX/20))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f')) # float
    plt.xlabel(r"y (cMpc/h)", fontweight='bold')
    plt.ylabel(r"x (cMpc/h)", fontweight='bold')
    plt.colorbar()
    plt.title(r'z-Projected {0}'.format(save_txt1))
    plt.savefig("{0}/{1}zProj{2}_{3}Log.pdf".format(config.NEXUS_DEST, config.DM_TYPE.upper(), save_txt2, config.SNAP))