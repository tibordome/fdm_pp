#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 15:33:36 2021

@author: tibor
"""

import numpy as np
from copy import deepcopy
from scipy import stats
from itertools import combinations
import math
from math import atan2,degrees
import make_grid_cic 
from scipy.fftpack import fftn, ifftn, fftfreq
from scipy.spatial.transform import Rotation as R
from sklearn.utils import resample
from scipy import integrate
import json
import itertools
from pynverse import inversefunc
from scipy.integrate import quad
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
from scipy import special
from math import isnan
import config
from config import makeGlobalSNAP, getA

def getMWDMFromFDM(m_FDM):
    """ Return WDM mass in keV for m_FDM in eV
    Assumes T convention, not T^2"""
    return 0.84*(m_FDM/(10**(-22)))**0.39

def H(a, HUBBLE, OMEGA_M):
    """ Return Hubble factor at scale factor a, for Hubble constant HUBBLE
    and dark matter fraction OMEGA_M"""
    return np.sqrt(HUBBLE**2*(OMEGA_M/a**3 + 1 - OMEGA_M))

def chiIntegrand(a, C, HUBBLE, OMEGA_M):
    """ The a-integrand in the definition of \chi(a)"""
    return C/(a**2*H(a, HUBBLE, OMEGA_M))

def getZDistroChiUnnorm(chi, z_med, beta, C, HUBBLE, OMEGA_M):
    """ Why Unnorm vs real one? A: Even with the Gamma function added to n(chi), since chi_max is not infinity,
    not even at LSS, integrating the function will not give exactly 1."""
    getchi = lambda z: quad(chiIntegrand, 1/(1+z), 1, args=(C, HUBBLE, OMEGA_M))[0]
    z = inversefunc(getchi, y_values = chi)
    return H(1/(z+1), HUBBLE, OMEGA_M)/C*beta*np.sqrt(2)/(z_med*special.gamma(3/beta))*(z*np.sqrt(2)/z_med)**2*np.exp(-(z*np.sqrt(2)/z_med)**beta)    

# Normalizing Z-Distros, can be called by multiple ranks
def normalizeZDistro(bin_edges, HUBBLE, OMEGA_M, C, CHI_H, Z_MED, BETA):
    """ Return normalized redshift-distribution n(z)"""
    getchi = lambda z: quad(chiIntegrand, 1/(1+z), 1, args=(C, HUBBLE, OMEGA_M))[0]
    
    def getZDistroZUnnorm(z, z_med, beta):
        return beta*np.sqrt(2)/(z_med*special.gamma(3/beta))*(z*np.sqrt(2)/z_med)**2*np.exp(-(z*np.sqrt(2)/z_med)**beta)
    
    norms = []
    for i in range(len(bin_edges)-1):
        norms.append(quad(getZDistroZUnnorm, inversefunc(getchi, y_values = bin_edges[i]), inversefunc(getchi, y_values = bin_edges[i+1]), args=(Z_MED, BETA))[0])
    def getZDistroZ(z, z_med, beta):
        chi_ = [getchi(z)]
        bin_ = np.digitize(chi_, bin_edges)[0]-1
        if bin_ == len(bin_edges) - 1: # Can happen since in float precision, maximal chi can be a tiny bit beyond bin_edges[-1]
            bin_ -= 1
        return getZDistroZUnnorm(z, z_med, beta)/norms[bin_]
    return getZDistroZ

def getTomEdges(CHI_VEC, Z_MED, BETA, TOM_BINS, C, HUBBLE, OMEGA_M):
    """ Return the edges of the tomographic bins, in units of comoving distance"""
    z_distro = np.zeros(CHI_VEC.shape[0], dtype = np.float32)
    for chi_ in range(CHI_VEC.shape[0]):
        z_distro[chi_] = getZDistroChiUnnorm(CHI_VEC[chi_], Z_MED, BETA, C, HUBBLE, OMEGA_M)
    bin_edges = weightedQuantile(CHI_VEC, np.linspace(0.0, 1.0, TOM_BINS + 1), sample_weight=z_distro)
    return bin_edges

def weightedQuantile(values, quantiles, sample_weight=None, 
                      values_sorted=False, old_style=True):
        """ Very close to numpy.percentile, but supports weights.
        NOTE: quantiles should be in [0, 1]!
        :param values: numpy.array with data
        :param quantiles: array-like with many quantiles needed
        :param sample_weight: array-like of the same length as `array`
        :param values_sorted: bool, if True, then will avoid sorting of
            initial array
        :param old_style: if True, will correct output to be consistent
            with numpy.percentile. True seems to be better 
            for finding tomographic bin edges.
        :return: numpy.array with computed quantiles.
        """
        values = np.array(values)
        quantiles = np.array(quantiles)
        if sample_weight is None:
            sample_weight = np.ones(len(values))
        sample_weight = np.array(sample_weight)
        assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
            'Quantiles should be in [0, 1]'
    
        if not values_sorted:
            sorter = np.argsort(values)
            values = values[sorter]
            sample_weight = sample_weight[sorter]
    
        weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
        if old_style:
            # To be convenient with numpy.percentile
            weighted_quantiles -= weighted_quantiles[0]
            weighted_quantiles /= weighted_quantiles[-1]
        else:
            weighted_quantiles /= np.sum(sample_weight)
        return np.interp(quantiles, weighted_quantiles, values)

def getHessianRealSpace(old_modes, dm_xyz, dm_masses, TIDAL_SMOOTH, OMEGA_M, A, CHI_H, angle=0.0):
    """ Calculates the full (also last axis non-truncated) real-space Hessian
    Note that dm_masses has to have equal entries, since the rotation matrix would mess around with it.
    Explicit Arguments:
    ------------
    old_modes: non-rotated modes (object centers) in original xyz-frame
    TIDAL_SMOOTH: Float or Double, smoothing scale
    Returns:
    ------------
    A (3, 3, N, N, N)-shaped array, encoding the real space Hessian of the smoothed grid, 
    with its first 2 indices (i and j) accessing the (i,j)-component of the Hessian
    new_modes: new modes (i.e. rotated, with respect to new coordinate frame)
    """
    
    # Real-space density field grid after active rotation
    if angle != 0.0:
        # 9 (if MULT = 1) boxes' direction vectors
        all_combs = [p for p in itertools.product(np.arange(-config.MULT,config.MULT+1), repeat=2)]
        buffer_dir = [np.array([x[0],x[1],0]) for x in all_combs]
        nb_jobs_to_do = len(buffer_dir)
        perrank = nb_jobs_to_do//size + (nb_jobs_to_do//size == 0)*1
        do_sth = rank <= nb_jobs_to_do-1
        count = 0
        if size <= nb_jobs_to_do:
            last = rank == size - 1 # Whether or not last process
        else:
            last = rank == nb_jobs_to_do - 1
        x_angle = np.empty(0, dtype = np.float32)
        y_angle = np.empty(0, dtype = np.float32)
        z_angle = np.empty(0, dtype = np.float32)
        center = np.array([1,1,0])*config.L_BOX/2 # This is the xy-center of the original [L,L,L] box.
        for box in range(rank*perrank, rank*perrank+do_sth*(perrank+last*(nb_jobs_to_do-(rank+1)*perrank))):
            # Box's direction vectors
            dir_ = buffer_dir[box]
            box_points = dm_xyz-center+config.L_BOX*dir_
            rot_points = np.hstack((np.reshape(math.cos(angle)*box_points[:,0]-math.sin(angle)*box_points[:,1],(box_points.shape[0],1)), np.reshape(math.sin(angle)*box_points[:,0]+math.cos(angle)*box_points[:,1],(box_points.shape[0],1)), np.reshape(box_points[:,2], (box_points.shape[0],1))))
            mask = np.logical_and(rot_points[:,0] <= config.L_BOX*config.MULT, np.logical_and(rot_points[:,0] > 0.0, np.logical_and(rot_points[:,1] <= config.L_BOX*config.MULT, rot_points[:,1] > 0.0)))
            x_angle = np.hstack((x_angle, np.float32(rot_points[mask,0])))
            y_angle = np.hstack((y_angle, np.float32(rot_points[mask,1])))
            z_angle = np.hstack((z_angle, np.float32(rot_points[mask,2])))
            count += mask[mask == True].shape[0]
        count_new = comm.gather(count, root=0)
        count_new = comm.bcast(count_new, root = 0)
        nb_dm_ptcs = np.sum(np.array(count_new))
        comm.Barrier()
        recvcounts = np.array(count_new)
        rdispls = np.zeros_like(recvcounts)
        for j in range(rdispls.shape[0]):
            rdispls[j] = np.sum(recvcounts[:j])
        dm_x_total = np.empty(nb_dm_ptcs, dtype = np.float32)
        dm_y_total = np.empty(nb_dm_ptcs, dtype = np.float32)
        dm_z_total = np.empty(nb_dm_ptcs, dtype = np.float32)
        comm.Gatherv(x_angle, [dm_x_total, recvcounts, rdispls, MPI.FLOAT], root = 0)
        comm.Gatherv(y_angle, [dm_y_total, recvcounts, rdispls, MPI.FLOAT], root = 0)
        comm.Gatherv(z_angle, [dm_z_total, recvcounts, rdispls, MPI.FLOAT], root = 0)
        xyz_angle = np.hstack((np.reshape(dm_x_total, (dm_x_total.shape[0],1)), np.reshape(dm_y_total, (dm_y_total.shape[0],1)), np.reshape(dm_z_total, (dm_z_total.shape[0],1))))
        
        new_modes = old_modes
        if rank == 0:
            # Find new halo mode positions
            new_modes = np.zeros_like(old_modes)
            for halo in range(old_modes.shape[0]):
                modes_points = np.empty(0, dtype = np.float32)
                for box in range(nb_jobs_to_do):
                    modes_points = np.hstack((modes_points, old_modes[halo]-center+config.L_BOX*buffer_dir[box]))
                modes_points = np.reshape(modes_points, ((modes_points.shape[0]//3,3)))
                rot_modes = np.hstack((np.reshape(math.cos(angle)*modes_points[:,0]-math.sin(angle)*modes_points[:,1],(modes_points.shape[0],1)), np.reshape(math.sin(angle)*modes_points[:,0]+math.cos(angle)*modes_points[:,1],(modes_points.shape[0],1)), np.reshape(modes_points[:,2], (modes_points.shape[0],1))))
                mask = np.logical_and(rot_modes[:,0] <= config.L_BOX*config.MULT, np.logical_and(rot_modes[:,0] > 0.0, np.logical_and(rot_modes[:,1] <= config.L_BOX*config.MULT, rot_modes[:,1] > 0.0)))
                modes_inside = rot_modes[mask]
                if modes_inside.shape[0] > 0:
                    center_ = np.array([1,1])*config.L_BOX*config.MULT/2 # Center of [L*Mult,L*Mult,L] cuboid.
                    dist_to_center_sq = (modes_inside[:,0]-center_[0])**2+(modes_inside[:,1]-center_[1])**2
                    new_modes[halo] = modes_inside[np.argmin(dist_to_center_sq)] # In case there multiple copies of same halo falling into cuboid of size L*Mult,L*Mult,L, choose the one closest to central z-axis (though it wouldn't matter which one is retained).
                else:
                    raise Exception("Your halo got lost.") # Will not happen if config.MULT >= 2
        
    else:
        if rank == 0:
            xyz_angle = dm_xyz
            new_modes = old_modes
    if rank == 0:
        
        # Parameters
        box_xy = config.L_BOX*(1 + (angle != 0.0)*config.MULT)
        box_z = config.L_BOX
        res_xy = config.N*(1 + (angle != 0.0)*config.MULT)
        res_z = config.N
        
        # Painting onto grid
        grid = make_grid_cic.makeGridWithCICPBCUnequal(xyz_angle[:,0].astype('float32'), xyz_angle[:,1].astype('float32'), xyz_angle[:,2].astype('float32'), np.ones_like(xyz_angle[:,0], dtype=np.float32)*np.float32(dm_masses[0]), box_xy, box_xy, box_z, res_xy, res_xy, res_z)
    
        # With FFTn, vectorized
        khess = np.zeros((res_xy, res_xy, res_z), dtype=np.complex128)
        rhess = np.zeros((3, 3, res_xy, res_xy, res_z), dtype=np.float32) # For res = 512, this is ~ 2 GBytes.
        
        # Fourier-space density field grid
        kgrid = fftn(grid).astype(np.complex128)
        k_xy = fftfreq(res_xy, box_xy/res_xy) # Has negative Nyquist frequency
        k_z = fftfreq(res_z, box_z/res_z) # Has negative Nyquist frequency
        kx,ky,kz = np.meshgrid(k_xy,k_xy,k_z, indexing='ij') # Cannot be cythonized
        k_mesh = kx,ky,kz # 'ij' since want (N1, N2, N3,...Nn)-shaped arrays, not (N2, N1, N3,...Nn) 
        k3d_squared = k_mesh[0]**2+k_mesh[1]**2+k_mesh[2]**2    
        k3d_squared[k3d_squared==0.0] = 1.0 # Set infinite wavelength-mode to 1. Just translation anyway.
        for i in range(3):
            for j in range(3):
                if i <= j:
                    khess = np.zeros((res_xy, res_xy, res_z), dtype=np.complex128)
                    khess = 4*np.pi**2*kgrid*np.exp(-k3d_squared*TIDAL_SMOOTH**2*4*np.pi**2/2)*TIDAL_SMOOTH**2*k_mesh[i]*k_mesh[j]/k3d_squared
                    rhess[i,j,:,:,:] = ifftn(khess).real.astype(np.float32)*3*OMEGA_M/(2*CHI_H**2*A)
                    if i != j: # The Hessian is symmetric
                        rhess[j,i,:,:,:] = rhess[i,j,:,:,:]
        return rhess, new_modes
    else:
        return None, None

def addBufferCells(xyz, L):
    """
    Implements PBC by glueing 8 "boxes" to the existing data
    Parameters
    ---
    xyz: (N,3) array, (x,y,z)-coordinates of the point distribution
    multiple: How thick should the slice (or square-based mini-pillar, or mini-cube) be?
    How many times the average particle spacing?
    to_be_extended: (N,)-arrays of any type kind that might want to be extended in a PBC fashion, e.g. mass array
    Returns
    ---
    (9*N,3) array: The spatial coordinates x, y, z, of the point distribution 
    extended by 8 boxes, each of size L (usually L = size of original box)
    """
    # Adding 8 slices to the existing data
    xyz_copy = deepcopy(xyz) # Recall that np.arrays are mutable
    
    # Let us start with the six boxes whose faces touch the ones of the given box
    # In x+ direction
    xyz_copy = np.vstack((xyz_copy, xyz+L*np.array([1,0,0])))
    # In x- direction
    xyz_copy = np.vstack((xyz_copy, xyz+L*np.array([-1,0,0])))
    # In y+ direction
    xyz_copy = np.vstack((xyz_copy, xyz+L*np.array([0,1,0])))
    # In y- direction
    xyz_copy = np.vstack((xyz_copy, xyz+L*np.array([0,-1,0])))
    # In x+y+ direction
    xyz_copy = np.vstack((xyz_copy, xyz+L*np.array([1,1,0])))
    # In x+y- direction
    xyz_copy = np.vstack((xyz_copy, xyz+L*np.array([1,-1,0])))
    # In x-y+ direction
    xyz_copy = np.vstack((xyz_copy, xyz+L*np.array([-1,1,0])))
    # In x-y- direction
    xyz_copy = np.vstack((xyz_copy, xyz+L*np.array([-1,-1,0])))
    
    return xyz_copy
    
def getStandardDeviation(x, y, m_est, b_est, nb_bins):
    """ Fetch the standard deviation in all nb_bins bins
    Returns:
    -------------
    bin_centers: 1D array of bin centers
    sd: 1D array of SDs"""
    bin_edges = stats.mstats.mquantiles(x, np.linspace(0.0, 1.0, nb_bins+1))
    bin_centers = []
    sd = []
    for bin_ in range(nb_bins):
        points = []
        for point in range(x.shape[0]):
            if x[point] < bin_edges[bin_+1] and x[point] > bin_edges[bin_]:
                points.append(point)
        sd_ = np.std(y[points]-b_est-m_est*x[points], ddof= 2)
        if points == []:
            sd.append(np.nan)
            bin_centers.append(np.nan)
        else:
            sd.append(sd_)
            bin_centers.append((bin_edges[bin_+1]+bin_edges[bin_])/2)
    return np.array(bin_centers), np.array(sd)
    
def linearRegression(x,y):
    """ Perform linear regression with standard formula on 2D point cloud
    Returns:
    m_est: estimate of the linear slope
    b_est: estimate of b in y = m*x + b
    delta_m_est: error estimate on m_est"""
    n = x.shape[0]
    x_mean = np.average(x)
    y_mean = np.average(y)
    m_est = np.sum((x-x_mean)*(y-y_mean))/np.sum((x-x_mean)**2)
    b_est = y_mean - x_mean
    sigma_y_est = np.sqrt(1/(n-2)*np.sum((y-b_est-m_est*x)**2)) # We loose 1 dof with m_est, another one with b_est.
    delta_m_est = np.sqrt(n/(n*np.sum(x**2)-(np.sum(x)**2)))*sigma_y_est
    return m_est, b_est, delta_m_est

def getKappa(h_cat, dm_xyz, dm_masses, dm_velxyz):
    """ Get kappa = e_rot / e_kin for all objects in h_cat"""
    kappa = []
    for halo in h_cat:
        if halo != []: 
            xyz = respectPBC(dm_xyz[halo])
            velxyz = dm_velxyz[halo]
            masses = dm_masses[halo]
            com = np.sum(xyz*np.reshape(masses, (masses.shape[0],1)), axis = 0)/masses.sum()
            com_vel = np.sum(velxyz*np.reshape(masses, (masses.shape[0],1)), axis = 0)/masses.sum()
            ang_mom_tot = np.zeros((3,), dtype=np.float32)
            for dm_ptc in range(xyz.shape[0]):
                ang_mom_tot += masses[dm_ptc]*np.cross(xyz[dm_ptc]-com, velxyz[dm_ptc]-com_vel)
            ang_mom_tot /= np.linalg.norm(ang_mom_tot) # Normalize total angular momentum vector
            rot_energy = 0.0
            for dm_ptc in range(xyz.shape[0]):
                rot_energy += 1/2*masses[dm_ptc]*(np.dot(masses[dm_ptc]*np.cross(xyz[dm_ptc]-com, velxyz[dm_ptc]-com_vel), ang_mom_tot)/(np.linalg.norm(xyz[dm_ptc]-com)-np.dot(xyz[dm_ptc]-com, ang_mom_tot)))**2
            kin_energy = 0.0
            for dm_ptc in range(xyz.shape[0]):
                kin_energy += 1/2*masses[dm_ptc]*np.dot(velxyz[dm_ptc]-com_vel,velxyz[dm_ptc]-com_vel)
            kappa.append(rot_energy/kin_energy)
    kappa = np.array(kappa)
    return kappa

def getTangent(pos_vec, samples, sampling_pos):
    """ Get the tangent vector representing fil spine of closest fil found in sampling
    Input:
    pos_vec: (3,), point of interest
    samples: list of sampling points in the format [x, y, z, fil_idx]
    sampling_pos: list of lists (for each fil) of sampling points in the format [x, y, z]"""
    
    node_affil1 = np.argmin((pos_vec[0]-samples[:,0])**2+(pos_vec[1]-samples[:,1])**2+(pos_vec[2]-samples[:,2])**2)
    fil_affil1 = np.int32(samples[node_affil1][3])
    samples_fil_affil1 = np.array([sampling_pos[fil_affil1][0][0], sampling_pos[fil_affil1][0][1], sampling_pos[fil_affil1][0][2]])
    for j in range(len(sampling_pos[fil_affil1])):
        samples_fil_affil1 = np.vstack((samples_fil_affil1, np.array([sampling_pos[fil_affil1][j][0], sampling_pos[fil_affil1][j][1], sampling_pos[fil_affil1][j][2]])))
    samples_fil_affil1 = np.delete(samples_fil_affil1,0,axis=0)
    
    # Remove node_affil1 point from samples_fil_affil1
    samples_fil_affil1 = np.delete(samples_fil_affil1,np.argmin((samples[node_affil1][0]-samples_fil_affil1[:,0])**2+(samples[node_affil1][1]-samples_fil_affil1[:,1])**2+(samples[node_affil1][2]-samples_fil_affil1[:,2])**2),axis=0)
    
    # Find node_affil2
    node_affil2 = np.argmin((pos_vec[0]-samples_fil_affil1[:,0])**2+(pos_vec[1]-samples_fil_affil1[:,1])**2+(pos_vec[2]-samples_fil_affil1[:,2])**2)
    return np.array([samples_fil_affil1[node_affil2][0]-samples[node_affil1][0], samples_fil_affil1[node_affil2][1]-samples[node_affil1][1], samples_fil_affil1[node_affil2][2]-samples[node_affil1][2]])

def findMode(xyz, masses, rad):
    """ Find mode of point distribution xyz
    Input:
    -------------
    xyz: (N,3)-float array
    rad: initial radius to consider away from COM of object"""
    com = np.sum(xyz*np.reshape(masses, (masses.shape[0],1)), axis = 0)/masses.sum()
    distances_all = np.linalg.norm(xyz-com,axis=1)
    xyz_constrain = xyz[distances_all < rad]
    masses_constrain = masses[distances_all < rad]
    if xyz_constrain.shape[0] < 5: # If only < 5 particles left, return
        return com
    else:
        rad *= 0.83 # Reduce radius by 17 %
        return findMode(xyz_constrain, masses_constrain, rad)

def respectPBC(xyz):
    """
    If point distro xyz has particles separated in any Cartesian direction
    by more than config.L_BOX/2, reflect those particles along config.L_BOX/2"""
    xyz_out = xyz.copy() # Otherwise changes would be reflected in outer scope (np.array is mutable).
    ref = 0 # Reference particle does not matter
    dist_x = abs(xyz_out[ref, 0]-xyz_out[:,0])
    xyz_out[:,0][dist_x > config.L_BOX/2] = config.L_BOX-xyz_out[:,0][dist_x > config.L_BOX/2] # Reflect x-xyz_outition along config.L_BOX/2
    dist_y = abs(xyz_out[ref, 1]-xyz_out[:,1])
    xyz_out[:,1][dist_y > config.L_BOX/2] = config.L_BOX-xyz_out[:,1][dist_y > config.L_BOX/2] # Reflect y-xyz_outition along config.L_BOX/2
    dist_z = abs(xyz_out[ref, 2]-xyz_out[:,2])
    xyz_out[:,2][dist_z > config.L_BOX/2] = config.L_BOX-xyz_out[:,2][dist_z > config.L_BOX/2] # Reflect z-xyz_outition along config.L_BOX/2
    return xyz_out

def getEpsilon(h_cat, dm_xyz, dm_masses, angle=0.0):
    """ Calculate the complex ellipticity from the shape tensor = centred (wrt mode) second mass moment tensor"""
    if rank == 0:
        eps = []
        rot_matrix = R.from_rotvec(angle * np.array([0, 0, 1])).as_matrix()
        for halo in h_cat:
            if halo != []:
                xyz = respectPBC(dm_xyz[halo])
                masses = dm_masses[halo]
                mode = findMode(xyz, masses, max((max(xyz[:,0])-min(xyz[:,0]), max(xyz[:,1])-min(xyz[:,1]), max(xyz[:,2])-min(xyz[:,2]))))
                xyz_new = np.zeros((xyz.shape[0],3))
                for i in range(xyz_new.shape[0]):
                    xyz_new[i] = np.dot(rot_matrix, xyz[i]-mode)
                shape_tensor = np.sum((masses)[:,np.newaxis,np.newaxis]*(np.matmul(xyz_new[:,:,np.newaxis],xyz_new[:,np.newaxis,:])),axis=0)/np.sum(masses)
                qxx = shape_tensor[0,0]
                qyy = shape_tensor[1,1]
                qxy = shape_tensor[0,1]
                eps.append((qxx-qyy)/(qxx+qyy) + complex(0,1)*2*qxy/(qxx+qyy))
        eps = np.array(eps)
        return eps
    else:
        return None

def getAllModes(h_cat, dm_xyz, dm_masses):
    """ Get mode of each object in the catalogue
    Returns:
    --------------
    modes: (# of objects, 3) float array"""
    if rank == 0:
        # Halo Modes
        modes = np.empty(0, dtype = np.float32)
        for halo in h_cat:
            if halo != []:
                xyz = respectPBC(dm_xyz[halo])
                masses = dm_masses[halo]
                modes = np.hstack((modes, findMode(xyz, masses, max((max(xyz[:,0])-min(xyz[:,0]), max(xyz[:,1])-min(xyz[:,1]), max(xyz[:,2])-min(xyz[:,2]))))))
        modes = np.reshape(modes, ((modes.shape[0]//3,3)))
        return modes
    else:
        return None
            
def getDist(x, y, L_BOX):
    """ Return Euclidian distance between 3D points x and y
    in periodic box of periodicity L_BOX"""

    dist_x = abs(x[0]-y[0])
    if dist_x > L_BOX/2:
        dist_x = L_BOX-dist_x
    dist_y = abs(x[1]-y[1])
    if dist_y > L_BOX/2:
        dist_y = L_BOX-dist_y
    dist_z = abs(x[2]-y[2])
    if dist_z > L_BOX/2:
        dist_z = L_BOX-dist_z  
    return np.sqrt(dist_x**2+dist_y**2+dist_z**2)


# Chi Square Test of Independence (Q we ask: Is the PDF independent of mass?)
def getChiSquare(mean_2D): # https://libguides.library.kent.edu/spss/chisquare
    """ Get chi square and p-value, assuming mean_2D[0,:] 
    and mean_2D[1,:] are independent data sets
    Input:
    -----------
    mean_2D: (2 = # mass groups, # mass or sep bins), spine of shaded curves, but 
             rescaled to be cell counts (or weighted cell counts), not PDF"""
    assert mean_2D.shape[0] == 2, \
            'Chi square test of independence for n != 2 makes no sense'
    chi_square = 0.0
    for group in range(mean_2D.shape[0]): 
        for d_bin in range(config.BINS_DSPINE):
            expect = mean_2D[group].sum()*mean_2D[:,d_bin].sum()/mean_2D.flatten().sum()
            if expect != 0.0:
                chi_square += (mean_2D[group, d_bin] - expect)**2/expect
    dof = (mean_2D.shape[0]-1)*(config.BINS_DSPINE-1)
    p_chi_square = 1 - stats.chi2.cdf(chi_square, dof) # p value for chi_square and dof dof
    return chi_square, p_chi_square, dof

def readDataGx():
    """ Read in all relevant Gx data"""
    
    if config.HALO_REGION == "Inner":
        with open('{0}/a_com_cat_local_fdm_gx_{1}.txt'.format(config.CAT_DEST, config.SNAP), 'r') as filehandle:
            a_com_cat_fdm = json.load(filehandle)
        d_fdm = np.loadtxt('{0}/d_local_fdm_gx_{1}.txt'.format(config.CAT_DEST, config.SNAP))
        q_fdm = np.loadtxt('{0}/q_local_fdm_gx_{1}.txt'.format(config.CAT_DEST, config.SNAP))
        s_fdm = np.loadtxt('{0}/s_local_fdm_gx_{1}.txt'.format(config.CAT_DEST, config.SNAP))
        # Dealing with the case of 1 gx
        if d_fdm.shape[0] == config.D_BINS+1:
            d_fdm = d_fdm.reshape(1, config.D_BINS+1)
            q_fdm = q_fdm.reshape(1, config.D_BINS+1)
            s_fdm = s_fdm.reshape(1, config.D_BINS+1)
        with open('{0}/gx_cat_local_fdm_{1}.txt'.format(config.CAT_DEST, config.SNAP), 'r') as filehandle:
            gx_cat_fdm = json.load(filehandle)
    else:
        assert config.HALO_REGION == "Full"
        with open('{0}/a_com_cat_overall_fdm_gx_{1}.txt'.format(config.CAT_DEST, config.SNAP), 'r') as filehandle:
            a_com_cat_fdm = json.load(filehandle)
        d_fdm = np.loadtxt('{0}/d_overall_fdm_gx_{1}.txt'.format(config.CAT_DEST, config.SNAP))
        d_fdm = d_fdm.reshape(d_fdm.shape[0], 1) # Has shape (number_of_gxs, 1)
        q_fdm = np.loadtxt('{0}/q_overall_fdm_gx_{1}.txt'.format(config.CAT_DEST, config.SNAP))
        q_fdm = q_fdm.reshape(q_fdm.shape[0], 1) # Has shape (number_of_gxs, 1)
        s_fdm = np.loadtxt('{0}/s_overall_fdm_gx_{1}.txt'.format(config.CAT_DEST, config.SNAP))
        s_fdm = s_fdm.reshape(s_fdm.shape[0], 1) # Has shape (number_of_gxs, 1)
        with open('{0}/gx_cat_overall_fdm_{1}.txt'.format(config.CAT_DEST, config.SNAP), 'r') as filehandle:
            gx_cat_fdm = json.load(filehandle)
    sh_masses_fdm = np.loadtxt('{0}/m_delta_fdm_dm_{1}.txt'.format(config.CAT_DEST, config.SNAP)) # Has shape (number_of_hs,)
    return a_com_cat_fdm, gx_cat_fdm, d_fdm, q_fdm, s_fdm, sh_masses_fdm

def readDataFDM(get_skeleton = False):
    """ Read in all relevant FDM data, with or without skeleton"""
    
    if config.HALO_REGION == "Inner":
        with open('{0}/a_com_cat_local_fdm_dm_{1}.txt'.format(config.CAT_DEST, config.SNAP), 'r') as filehandle:
            a_com_cat = json.load(filehandle)
        d = np.loadtxt('{0}/d_local_fdm_dm_{1}.txt'.format(config.CAT_DEST, config.SNAP)) # Has shape (number_of_halos, config.D_BINS+1)
        q = np.loadtxt('{0}/q_local_fdm_dm_{1}.txt'.format(config.CAT_DEST, config.SNAP)) # Has shape (number_of_halos, config.D_BINS+1)
        s = np.loadtxt('{0}/s_local_fdm_dm_{1}.txt'.format(config.CAT_DEST, config.SNAP)) # Has shape (number_of_halos, config.D_BINS+1)
        # Dealing with the case of 1 halo
        if d.shape[0] == config.D_BINS+1:
            d = d.reshape(1, config.D_BINS+1)
            q = q.reshape(1, config.D_BINS+1)
            s = s.reshape(1, config.D_BINS+1)
        major_full = np.loadtxt('{0}/major_local_fdm_dm_{1}.txt'.format(config.CAT_DEST, config.SNAP))
        if major_full.ndim == 2: 
            major_full = major_full.reshape(major_full.shape[0], major_full.shape[1]//3, 3) # Has shape (number_of_halos, config.D_BINS+1, 3)
        else:
            if major_full.shape[0] == (config.D_BINS+1)*3: # This case is when there is only 1 halo (np.savetxt loses (1,config.D_BINS+1) array shape --> (config.D_BINS+1,))
                major_full = major_full.reshape(1, config.D_BINS+1, 3)
        with open('{0}/sh_cat_local_fdm_{1}.txt'.format(config.CAT_DEST, config.SNAP), 'r') as filehandle:
            h_cat = json.load(filehandle)
    else:
        assert config.HALO_REGION == "Full"
        with open('{0}/a_com_cat_overall_fdm_dm_{1}.txt'.format(config.CAT_DEST, config.SNAP), 'r') as filehandle:
            a_com_cat = json.load(filehandle)
        d = np.loadtxt('{0}/d_overall_fdm_dm_{1}.txt'.format(config.CAT_DEST, config.SNAP)) # Has shape (number_of_halos, )
        d = d.reshape(d.shape[0], 1) # Has shape (number_of_halos, 1)
        q = np.loadtxt('{0}/q_overall_fdm_dm_{1}.txt'.format(config.CAT_DEST, config.SNAP)) # Has shape (number_of_halos, )
        q = q.reshape(q.shape[0], 1) # Has shape (number_of_halos, 1)
        s = np.loadtxt('{0}/s_overall_fdm_dm_{1}.txt'.format(config.CAT_DEST, config.SNAP)) # Has shape (number_of_halos, )
        s = s.reshape(s.shape[0], 1) # Has shape (number_of_halos, 1)
        major_full = np.loadtxt('{0}/major_overall_fdm_dm_{1}.txt'.format(config.CAT_DEST, config.SNAP))
        if major_full.ndim == 2:
            major_full = major_full.reshape(major_full.shape[0], major_full.shape[1]//3, 3) # Has shape (number_of_halos, 1, 3)
        else:
            if major_full.shape[0] == 3:
                major_full = major_full.reshape(1, 1, 3)
        with open('{0}/sh_cat_overall_fdm_{1}.txt'.format(config.CAT_DEST, config.SNAP), 'r') as filehandle:
            h_cat = json.load(filehandle)
    rdelta = np.loadtxt('{0}/r_delta_fdm_dm_{1}.txt'.format(config.CAT_DEST, config.SNAP)) # Has shape (number_of_hs,)
    sh_masses = np.loadtxt('{0}/m_delta_fdm_dm_{1}.txt'.format(config.CAT_DEST, config.SNAP)) # Has shape (number_of_hs,)
    if get_skeleton == True:
        with open('{0}/sampling_pos_fdm_{1}.txt'.format(config.SKELETON_DEST, config.SNAP), 'r') as filehandle:
            sampling_pos = json.load(filehandle)
        samples = np.loadtxt('{0}/samples_fdm_{1}.txt'.format(config.SKELETON_DEST, config.SNAP))
        return a_com_cat, h_cat, d, q, s, major_full, rdelta, sh_masses, sampling_pos, samples
    else:
        return a_com_cat, h_cat, d, q, s, major_full, rdelta, sh_masses
    
def readDataVDispFDM():
    """ Read in all relevant DM vel disp data for one DM scenario"""
    
    q = np.loadtxt('{0}/q_vdisp_fdm_dm_{1}.txt'.format(config.CAT_DEST, config.SNAP)) # Has shape (number_of_shs, 1)
    q = q.reshape(q.shape[0], 1) # Has shape (number_of_shs, 1)
    s = np.loadtxt('{0}/s_vdisp_fdm_dm_{1}.txt'.format(config.CAT_DEST, config.SNAP)) # Has shape (number_of_shs, 1)
    s = s.reshape(s.shape[0], 1) # Has shape (number_of_shs, 1)
    major_full = np.loadtxt('{0}/major_vdisp_fdm_dm_{1}.txt'.format(config.CAT_DEST, config.SNAP))
    if major_full.ndim == 2:
        major_full = major_full.reshape(major_full.shape[0], major_full.shape[1]//3, 3) # Has shape (number_of_shs, 1, 3)
    else:
        if major_full.shape[0] == 3:
            major_full = major_full.reshape(1, 1, 3)
    with open('{0}/sh_cat_overall_fdm_dm_{1}.txt'.format(config.CAT_DEST, config.SNAP), 'r') as filehandle:
        h_cat = json.load(filehandle)
    return h_cat, q, s, major_full

def getZ(snaps, start_time):
    """ Returns redshifts from list of snaps"""
    z = np.zeros((len(snaps),), dtype = np.float32)
    for i, snap in enumerate(snaps):
        makeGlobalSNAP(snap, start_time)
        z[i] = getA()**(-1)-1
    return z

def assembleDataFDM(a_com_cat, h_cat, d, q, s, major_full, sh_masses):
    """ Assemble FDM data"""
    
    sh_masses_surv = np.empty(0, dtype=np.float32)
    for sh in range(len(h_cat)):
        if h_cat[sh] != []:
            sh_masses_surv = np.hstack((sh_masses_surv, sh_masses[sh]))
    assert sh_masses_surv.shape[0] == len(a_com_cat)
    
    sh_com = []
    for sh in range(len(a_com_cat)):
        sh_com.append(np.array([a_com_cat[sh][3], a_com_cat[sh][4], a_com_cat[sh][5]]))
    sh_com_arr = np.array(sh_com) # Has shape (number_of_shs, 3)
    
    if config.HALO_REGION == "Full":
        idx = np.array([np.int32(x) for x in list(np.ones((d.shape[0],))*(-1))])
    else:
        assert config.HALO_REGION == "Inner"
        idx = np.zeros((d.shape[0],), dtype = np.int32)
        for sh in range(idx.shape[0]):
            idx[sh] = np.argmin(abs(d[sh] - d[sh,-int(config.D_LOGEND/((config.D_LOGEND-config.D_LOGSTART)/config.D_BINS))-1]*0.15))
    major = np.zeros((len(a_com_cat), 3))
    for sh in range(len(a_com_cat)):
        major[sh] = np.array([major_full[sh, idx[sh], 0], major_full[sh, idx[sh], 1], major_full[sh, idx[sh], 2]])
    
    t = np.zeros((d.shape[0],))
    for sh in range(idx.shape[0]):
        t[sh] = (1-q[sh,idx[sh]]**2)/(1-s[sh,idx[sh]]**2) # Triaxiality
    t = np.nan_to_num(t)
    
    return sh_masses_surv, sh_com_arr, idx, major, t

def assembleDataGx(gx_cat_fdm, a_com_cat_fdm, q_fdm, s_fdm, sh_masses_fdm):
    """ Assemble Gx data for all gxs whose triaxility
    is in the window specified by config.T_CUT_LOW and config.T_CUT_HIGH
    Note that for gxs, there is no shape determination success related filtering.
    Returns:
    sh_masses_surv: (# gxs,) 1D float array, masses of shs
    gx_com_arr_fdm: (# gxs, 3) float array, COMs of gxs
    idx_x: (# gxs,) 1D int array, ellipsoidal radius index (HALO_REGION-dependent) 
    major_fdm: (# gxs, 3) float array, major axis of gxs
    t_fdm: (# gxs,) float array, triaxialities of gxs (after applying T cuts)"""
    
    sh_masses_surv = np.empty(0, dtype=np.float32)
    for sh in range(len(gx_cat_fdm)):
        if gx_cat_fdm[sh] != []:
            sh_masses_surv = np.hstack((sh_masses_surv, sh_masses_fdm[sh]))
            
    assert sh_masses_surv.shape[0] == len(a_com_cat_fdm)
            
    gx_com_fdm = []
    for gx in range(len(a_com_cat_fdm)):
        gx_com_fdm.append(np.array([a_com_cat_fdm[gx][3], a_com_cat_fdm[gx][4], a_com_cat_fdm[gx][5]]))
    gx_com_arr_fdm = np.array(gx_com_fdm) # Has shape (number_of_gxs, 3)
    gx_com_fdm = list(gx_com_arr_fdm)
    
    major_fdm = np.zeros((len(a_com_cat_fdm), 3))
    for gx in range(len(a_com_cat_fdm)):
        major_fdm[gx] = np.array([a_com_cat_fdm[gx][0], a_com_cat_fdm[gx][1], a_com_cat_fdm[gx][2]])
        
    t_fdm = np.zeros((q_fdm.shape[0],))
    for gx in range(q_fdm.shape[0]):
        t_fdm[gx] = (1-q_fdm[gx,-1]**2)/(1-s_fdm[gx,-1]**2) # Triaxiality   
    t_fdm = np.nan_to_num(t_fdm)
    
    if config.T_CUT_LOW != 0.0:
        sh_masses_surv = sh_masses_surv[t_fdm > config.T_CUT_LOW]
        gx_com_arr_fdm = gx_com_arr_fdm[t_fdm > config.T_CUT_LOW]
        major_fdm = major_fdm[t_fdm > config.T_CUT_LOW]
        t_fdm = t_fdm[t_fdm > config.T_CUT_LOW]
        
    if config.T_CUT_HIGH != 1.0:
        sh_masses_surv = sh_masses_surv[t_fdm < config.T_CUT_HIGH]
        gx_com_arr_fdm = gx_com_arr_fdm[t_fdm < config.T_CUT_HIGH]
        major_fdm = major_fdm[t_fdm < config.T_CUT_HIGH]
        t_fdm = t_fdm[t_fdm < config.T_CUT_HIGH]
    
    return sh_masses_surv, gx_com_arr_fdm, major_fdm, t_fdm

def getPDF(seps, d_bins):
    """ Get PDF of distance distribution from nearest skeleton
    Input:
    ------------
    seps: 1D float, all perpendicular distances to closest fil
    d_bins: edges of all distance bins"""
    seps_bins = [[] for i in range(config.BINS_DSPINE)]
    for dm_halo in range(len(seps)):
        closest_idx = (np.abs(d_bins - seps[dm_halo])).argmin() # Determine which point in d_bins is closest
        seps_bins[closest_idx].append(seps[dm_halo])
    
    mean = np.array([len(z) for z in seps_bins])
    stand_err = getStdErrSeps(seps, d_bins)
    is_negative = np.min(np.log10(d_bins)) < 0.0
    norm = integrate.trapz(mean, np.log10(d_bins) - is_negative*np.min(np.log10(d_bins))) # Returns total area under mean  
    return mean, stand_err, norm


def getMeanOrMedianAndError(y):
    """Return mean (if ERROR_METHOD == "bootstrap" or "SEM") or median
    (if ERROR_METHOD == "median_quantile") and the +- 1 sigma error attached
    Input:
    ----------
    y: 1D float array, data"""
    if config.ERROR_METHOD == "bootstrap":
        mean_median = np.array([np.average(z) if z != [] else np.nan for z in y])
        mean_l = [[] for i in range(len(y))]
        err_low = np.empty(len(y), dtype = np.float32)
        err_high = np.empty(len(y), dtype = np.float32)
        for random_state in range(config.N_REAL):
            for d_bin in range(len(y)):
                boot = resample(y[d_bin], replace=True, n_samples=len(y[d_bin]), random_state=random_state)
                mean_l[d_bin].append(np.average(boot))
        for d_bin in range(len(y)):
            err_low[d_bin] = np.std(mean_l[d_bin], ddof=1) # Says thestatsgeek.com
            err_high[d_bin] = err_low[d_bin]
    elif config.ERROR_METHOD == "SEM":
        mean_median = np.array([np.average(z) if z != [] else np.nan for z in y])
        err_low = np.array([np.std(z, ddof=1)/(np.sqrt(len(z))) for z in y])
        err_high = err_low
    else:
        assert config.ERROR_METHOD == "median_quantile"
        mean_median = np.array([np.median(z) if z != [] else np.nan for z in y])
        err_low = np.array([np.quantile(np.array(z), 0.25)/(np.sqrt(len(z))) if z != [] else np.nan for z in y])
        err_high = np.array([np.quantile(np.array(z), 0.75)/(np.sqrt(len(z))) if z != [] else np.nan for z in y])
    return mean_median, err_low, err_high

def getStdErrSeps(seps, d_bins):
    """ Get StdErr on the seps via overall bootstrapping, not just in each bin"""
    stand_err = np.empty(config.BINS_DSPINE, dtype = np.float32)
    mean_l = [[] for i in range(config.BINS_DSPINE)]
    for random_state in range(config.N_REAL):
        boot = resample(seps, replace=True, n_samples=len(seps), random_state=random_state)
        seps_bins_tmp = [[] for i in range(config.BINS_DSPINE)]
        for dm_halo in range(len(boot)):
            closest_idx = (np.abs(d_bins - boot[dm_halo])).argmin() # Determine which point in d_bins is closest
            seps_bins_tmp[closest_idx].append(boot[dm_halo])
        for d_bin in range(config.BINS_DSPINE):
            mean_l[d_bin].append(len(seps_bins_tmp[d_bin]))
    for d_bin in range(config.BINS_DSPINE):
        stand_err[d_bin] = np.std(mean_l[d_bin], ddof=1) # Says thestatsgeek.com
    return stand_err

def getStdErrSepsWeights(seps, d_bins, weights):
    """ Similar to getStdErrSeps but with weights attached to seps, e.g. from BG density"""
    stand_err = np.empty(config.BINS_DSPINE, dtype = np.float32)
    mean_l = [[] for i in range(config.BINS_DSPINE)]
    for random_state in range(config.N_REAL):
        boot = resample(seps, replace=True, n_samples=len(seps), random_state=random_state)
        boot_weights = resample(weights, replace=True, n_samples=len(seps), random_state=random_state)
        weights_bins_tmp = [[] for i in range(config.BINS_DSPINE)]
        for dm_halo in range(len(boot)):
            closest_idx = (np.abs(d_bins - boot[dm_halo])).argmin() # Determine which point in d_bins is closest
            weights_bins_tmp[closest_idx].append(boot_weights[dm_halo])
        for d_bin in range(config.BINS_DSPINE):
            if weights_bins_tmp[d_bin] != []:
                mean_l[d_bin].append(np.array(weights_bins_tmp[d_bin]).sum())
    for d_bin in range(config.BINS_DSPINE):
        if mean_l[d_bin] != []:
            stand_err[d_bin] = np.std(mean_l[d_bin], ddof=1) # Says thestatsgeek.com
    return stand_err

def sortSeparations(halo_com_arr):
    """ Find all pair distances between every object found in halo_com_arr
    and sort them
    Returns:
    ------------
    all_pairs_seps: list of tuples in the format (halo 1 idx, halo 2 idx, sep)
    seps: same as all_pairs_seps but only if sep < config.L_BOX / 2"""
    all_pairs = list((i,j) for ((i,_),(j,_)) in combinations(enumerate(halo_com_arr), 2))
    seps = []
    all_pairs_seps = []
    
    nb_jobs_to_do = len(all_pairs)
    perrank = nb_jobs_to_do//size + (nb_jobs_to_do//size == 0)*1
    do_sth = rank <= nb_jobs_to_do-1
    if size <= nb_jobs_to_do:
        last = rank == size - 1 # Whether or not last process
    else:
        last = rank == nb_jobs_to_do - 1
    count = 0
    count_raw = 0
    comm.Barrier()
    for pair in range(rank*perrank, rank*perrank+do_sth*(perrank+last*(nb_jobs_to_do-(rank+1)*perrank))):
        dist_x = abs(halo_com_arr[all_pairs[pair][0], 0]-halo_com_arr[all_pairs[pair][1], 0])
        if dist_x > config.L_BOX/2: # Respecting PBCs
            dist_x = config.L_BOX-dist_x
        dist_y = abs(halo_com_arr[all_pairs[pair][0], 1]-halo_com_arr[all_pairs[pair][1], 1])
        if dist_y > config.L_BOX/2:
            dist_y = config.L_BOX-dist_y
        dist_z = abs(halo_com_arr[all_pairs[pair][0], 2]-halo_com_arr[all_pairs[pair][1], 2])
        if dist_z > config.L_BOX/2:
            dist_z = config.L_BOX-dist_z
        all_pairs_seps.append((all_pairs[pair][0], all_pairs[pair][1], np.float32(np.sqrt(dist_x**2+dist_y**2+dist_z**2))))
        count_raw += 1
        if np.sqrt(dist_x**2+dist_y**2+dist_z**2) < config.L_BOX/2:
            seps.append((all_pairs[pair][0], all_pairs[pair][1], np.float32(np.sqrt(dist_x**2+dist_y**2+dist_z**2))))
            count += 1
    count_new = comm.gather(count, root=0) 
    count_raw_new = comm.gather(count_raw, root=0) 
    seps = comm.gather(seps, root=0)
    all_pairs_seps = comm.gather(all_pairs_seps, root=0)
    if rank == 0:
        seps = [seps[i][j] for i in range(size) for j in range(count_new[i])]
        all_pairs_seps = [all_pairs_seps[i][j] for i in range(size) for j in range(count_raw_new[i])]
    seps = comm.bcast(seps, root = 0)
    all_pairs_seps = comm.bcast(all_pairs_seps, root = 0)
    
    return all_pairs_seps, seps

def getNNPairs(halo_com_arr, all_pairs_seps):
    """ Return the distance between each halo found in halo_com_arr
    and its NN. Recall that all_pairs_seps is a list of 3-tuples,
    containing all (not just for sep > config.L_BOX/2) pairs."""
    NN_seps = []
    
    # NN analysis
    nb_jobs_to_do = len(list(halo_com_arr))
    nb_halos = nb_jobs_to_do
    perrank = nb_jobs_to_do//size + (nb_jobs_to_do//size == 0)*1
    do_sth = rank <= nb_jobs_to_do-1
    if size <= nb_jobs_to_do:
        last = rank == size - 1 # Whether or not last process
    else:
        last = rank == nb_jobs_to_do - 1
    count = 0
    comm.Barrier()
    for halo in range(rank*perrank, rank*perrank+do_sth*(perrank+last*(nb_jobs_to_do-(rank+1)*perrank))):
        pairs_seps = []
        # Add those pairs that are showing up as (halo, halo+1), ..., (halo, nb_halos-1)
        if halo < nb_halos - 1: # I.e. for halo = nb_halos - 1 (max), we do not have any contribution from here 
            pairs_seps = all_pairs_seps[int(round(halo/2*(2*nb_halos-halo-1))):int(round(halo/2*(2*nb_halos-halo-1)))+nb_halos-halo-1]
        # Add those pairs that are showing up earlier in seps_raw as (0, halo), ..., (halo-1,halo)
        pairs_seps += [all_pairs_seps[int(round(n/2*(2*nb_halos-n-1))) + halo - 1 - n] for n in range(halo)]
        if pairs_seps == []: # Can only happen if # halos is 1
            continue
        # Find NN
        argmin = np.argmin(np.array(pairs_seps)[:,2]) # np.array is important to transform list of tuples into array
        min_ = pairs_seps[argmin][2]
        if min_ < config.L_BOX/2:
            if pairs_seps[argmin][1] == halo:
                NN_seps.append((halo, pairs_seps[argmin][0], min_))
                count += 1
            if pairs_seps[argmin][0] == halo:
                NN_seps.append((halo, pairs_seps[argmin][1], min_))
                count += 1
    count_new = comm.gather(count, root=0) 
    NN_seps = comm.gather(NN_seps, root=0)
    if rank == 0:
        NN_seps = [NN_seps[i][j] for i in range(size) for j in range(count_new[i])]
    NN_seps = comm.bcast(NN_seps, root = 0)
    return NN_seps
        
def getSSOrSP(seps, NN_seps, major, halo_com_arr, get_ss = False):
    """ Get shape-shape or shape-position alignment, depending on get_ss.
    Only rank 0 will have result."""
    
    if seps == []:
        x = np.linspace(0, 1.0, config.MEAN_BINS)
        mean_median = np.empty(config.MEAN_BINS)*np.nan
        err_low = np.empty(config.MEAN_BINS)*np.nan
        err_high = np.empty(config.MEAN_BINS)*np.nan
        mean_median_nn = np.empty(config.MEAN_BINS)*np.nan
        err_low_nn = np.empty(config.MEAN_BINS)*np.nan
        err_high_nn = np.empty(config.MEAN_BINS)*np.nan
    else:
        x = np.linspace(np.min(seps), np.max(seps), config.MEAN_BINS)
        
        y = [[] for i in range(x.shape[0])] # List of lists (x.shape[0] many, indexable) of ?
        nb_jobs_to_do = len(seps)
        perrank = nb_jobs_to_do//size + (nb_jobs_to_do//size == 0)*1
        do_sth = rank <= nb_jobs_to_do-1
        if size <= nb_jobs_to_do:
            last = rank == size - 1 # Whether or not last process
        else:
            last = rank == nb_jobs_to_do - 1
        comm.Barrier()
        count = [0 for i in range(x.shape[0])] # List of (x.shape[0] many, indexable) numbers
        if get_ss == False:
            for pair in range(rank*perrank, rank*perrank+do_sth*(perrank+last*(nb_jobs_to_do-(rank+1)*perrank))):
                closest_idx = (np.abs(x - seps[pair][2])).argmin() # Determine which point in x is closest
                y[closest_idx].append(abs(np.dot(major[seps[pair][0]], halo_com_arr[seps[pair][0]]- halo_com_arr[seps[pair][1]]))/(np.linalg.norm(halo_com_arr[seps[pair][0]]- halo_com_arr[seps[pair][1]])*np.linalg.norm(major[seps[pair][0]])))
                y[closest_idx].append(abs(np.dot(major[seps[pair][1]], halo_com_arr[seps[pair][1]]- halo_com_arr[seps[pair][0]]))/(np.linalg.norm(halo_com_arr[seps[pair][1]]- halo_com_arr[seps[pair][0]])*np.linalg.norm(major[seps[pair][1]])))
                count[closest_idx] += 2
        else:
            for pair in range(rank*perrank, rank*perrank+do_sth*(perrank+last*(nb_jobs_to_do-(rank+1)*perrank))):
                closest_idx = (np.abs(x - seps[pair][2])).argmin() # Determine which point in x is closest
                y[closest_idx].append(abs(np.dot(major[seps[pair][0]], major[seps[pair][1]]))/(np.linalg.norm(major[seps[pair][0]])*np.linalg.norm(major[seps[pair][1]])))
                count[closest_idx] += 1
        count_new = comm.gather(count, root=0) # List of lists (size many, indexable) of (x.shape[0] many, indexable) numbers
        y = comm.gather(y, root=0) # List of lists (size many, indexable) of lists (x.shape[0] many, indexable) of (? many, indexable) numbers
        if rank == 0:
            y = [[y[i][j][k] for i in range(size) for k in range(count_new[i][j])] for j in range(x.shape[0])]
            y = [[z for z in x if not np.isnan(z)] for x in y]
            mean_median, err_low, err_high = getMeanOrMedianAndError(y)
        else:
            mean_median = None
            err_low = None
            err_high = None
        y_nn = [[] for i in range(x.shape[0])] # List of lists (x.shape[0] many, indexable) of ?
        nb_jobs_to_do = len(NN_seps)
        perrank = nb_jobs_to_do//size + (nb_jobs_to_do//size == 0)*1
        do_sth = rank <= nb_jobs_to_do-1
        if size <= nb_jobs_to_do:
            last = rank == size - 1 # Whether or not last process
        else:
            last = rank == nb_jobs_to_do - 1
        comm.Barrier()
        count = [0 for i in range(x.shape[0])] # List of (x.shape[0] many, indexable) numbers
        if get_ss == False:
            for nn in range(rank*perrank, rank*perrank+do_sth*(perrank+last*(nb_jobs_to_do-(rank+1)*perrank))):
                closest_idx = (np.abs(x - NN_seps[nn][2])).argmin() # Determine which point in x is closest
                y_nn[closest_idx].append(abs(np.dot(major[NN_seps[nn][0]], halo_com_arr[NN_seps[nn][0]]- halo_com_arr[NN_seps[nn][1]]))/(np.linalg.norm(halo_com_arr[NN_seps[nn][0]]- halo_com_arr[NN_seps[nn][1]])*np.linalg.norm(major[NN_seps[nn][0]])))
                y_nn[closest_idx].append(abs(np.dot(major[NN_seps[nn][1]], halo_com_arr[NN_seps[nn][1]]- halo_com_arr[NN_seps[nn][0]]))/(np.linalg.norm(halo_com_arr[NN_seps[nn][1]]- halo_com_arr[NN_seps[nn][0]])*np.linalg.norm(major[NN_seps[nn][1]])))
                count[closest_idx] += 2
        else:
            for nn in range(rank*perrank, rank*perrank+do_sth*(perrank+last*(nb_jobs_to_do-(rank+1)*perrank))):
                closest_idx = (np.abs(x - NN_seps[nn][2])).argmin() # Determine which point in x is closest
                y_nn[closest_idx].append(abs(np.dot(major[NN_seps[nn][0]], major[NN_seps[nn][1]]))/(np.linalg.norm(major[NN_seps[nn][0]])*np.linalg.norm(major[NN_seps[nn][1]])))
                count[closest_idx] += 1
        count_new = comm.gather(count, root=0) # List of lists (size many, indexable) of (x.shape[0] many, indexable) numbers
        y_nn = comm.gather(y_nn, root=0) # List of lists (size many, indexable) of lists (x.shape[0] many, indexable) of (? many, indexable) numbers
        if rank == 0:
            y_nn = [[y_nn[i][j][k] for i in range(size) for k in range(count_new[i][j])] for j in range(x.shape[0])]
            y_nn = [[z for z in x if not np.isnan(z)] for x in y_nn]
            mean_median_nn, err_low_nn, err_high_nn = getMeanOrMedianAndError(y_nn)
        else:
            mean_median_nn = None
            err_low_nn = None
            err_high_nn = None
    
    return x, mean_median, err_low, err_high, mean_median_nn, err_low_nn, err_high_nn 

def getSSOrSPRaw(seps, major, halo_com_arr, get_ss = False):
    """ Similar to getSSOrSP, but only returns SS/SP values in 1D float,
    primarily used for histogram purposes.
    Only rank 0 will have result."""
    
    if seps == []:
        alignment = list(np.empty(config.MEAN_BINS)*np.nan)
    else:
        alignment = []
        
        nb_jobs_to_do = len(seps)
        perrank = nb_jobs_to_do//size + (nb_jobs_to_do//size == 0)*1
        do_sth = rank <= nb_jobs_to_do-1
        if size <= nb_jobs_to_do:
            last = rank == size - 1 # Whether or not last process
        else:
            last = rank == nb_jobs_to_do - 1
        count = 0
        comm.Barrier()
        if get_ss == False:
            for run in range(rank*perrank, rank*perrank+do_sth*(perrank+last*(nb_jobs_to_do-(rank+1)*perrank))):
                alignment.append(abs(np.dot(major[seps[run][0]], halo_com_arr[seps[run][0]]- halo_com_arr[seps[run][1]]))/(np.linalg.norm(halo_com_arr[seps[run][0]]- halo_com_arr[seps[run][1]])*np.linalg.norm(major[seps[run][0]]))) 
                alignment.append(abs(np.dot(major[seps[run][1]], halo_com_arr[seps[run][1]]- halo_com_arr[seps[run][0]]))/(np.linalg.norm(halo_com_arr[seps[run][1]]- halo_com_arr[seps[run][0]])*np.linalg.norm(major[seps[run][1]]))) 
                count += 2
        else:
            for run in range(rank*perrank, rank*perrank+do_sth*(perrank+last*(nb_jobs_to_do-(rank+1)*perrank))):
                alignment.append(abs(np.dot(major[seps[run][0]], major[seps[run][1]]))/(np.linalg.norm(major[seps[run][0]])*np.linalg.norm(major[seps[run][1]]))) 
                count += 1
            
        count_new = comm.gather(count, root=0)
        alignment = comm.gather(alignment, root=0)
        if rank == 0:
            alignment = [alignment[i][j] for i in range(size) for j in range(count_new[i])]
        
    return alignment

def getSSOrSPRawRSplit(seps, major, halo_com_arr, max_min_r, group, get_ss = False):
    """ Similar to getSSOrSPRaw, but with R splitting.
    Only rank 0 will have result."""
    
    if seps == []:
        alignment = list(np.empty(config.MEAN_BINS)*np.nan)
    else:
        alignment = []
        nb_jobs_to_do = len(seps)
        perrank = nb_jobs_to_do//size + (nb_jobs_to_do//size == 0)*1
        do_sth = rank <= nb_jobs_to_do-1
        if size <= nb_jobs_to_do:
            last = rank == size - 1 # Whether or not last process
        else:
            last = rank == nb_jobs_to_do - 1
        count = 0
        comm.Barrier()
        if get_ss == False:
            for run in range(rank*perrank, rank*perrank+do_sth*(perrank+last*(nb_jobs_to_do-(rank+1)*perrank))):
                if seps[run][2] >= max_min_r[group] and seps[run][2] <= max_min_r[group+1]:
                    alignment.append(abs(np.dot(major[seps[run][0]], halo_com_arr[seps[run][0]]- halo_com_arr[seps[run][1]]))/(np.linalg.norm(halo_com_arr[seps[run][0]]- halo_com_arr[seps[run][1]])*np.linalg.norm(major[seps[run][0]]))) 
                    alignment.append(abs(np.dot(major[seps[run][1]], halo_com_arr[seps[run][1]]- halo_com_arr[seps[run][0]]))/(np.linalg.norm(halo_com_arr[seps[run][1]]- halo_com_arr[seps[run][0]])*np.linalg.norm(major[seps[run][1]]))) 
                    count += 2
        else:
            for run in range(rank*perrank, rank*perrank+do_sth*(perrank+last*(nb_jobs_to_do-(rank+1)*perrank))):
                if seps[run][2] >= max_min_r[group] and seps[run][2] <= max_min_r[group+1]:
                    alignment.append(abs(np.dot(major[seps[run][0]], major[seps[run][1]]))/(np.linalg.norm(major[seps[run][0]])*np.linalg.norm(major[seps[run][1]]))) 
                    count += 1
        count_new = comm.gather(count, root=0) 
        alignment = comm.gather(alignment, root=0)
        if rank == 0:
            alignment = [alignment[i][j] for i in range(size) for j in range(count_new[i])]
    return alignment

def getProfile(R, d, param_interest):
    """ Get average profile for param_interest (which is defined at all values of d)
    at all elliptical radii R"""
    y = [[] for i in range(config.R_BIN+1)]
    for obj in range(param_interest.shape[0]):
        for rad in range(config.D_BINS+1):
            closest_idx = (np.abs(R - d[obj,rad]/d[obj,-int(config.D_LOGEND/((config.D_LOGEND-config.D_LOGSTART)/config.D_BINS))-1])).argmin() # Determine which point in R is closest
            if isnan(param_interest[obj][rad]) or np.log10(d[obj][rad]/d[obj,-int(config.D_LOGEND/((config.D_LOGEND-config.D_LOGSTART)/config.D_BINS))-1]) > config.R_LOGEND:
                continue
            else:
                y[closest_idx].append(param_interest[obj][rad])
    mean, err_low, err_high = getMeanOrMedianAndError(y)
    return mean, err_low, err_high

def getProfileMs(R, d, idx_groups, group, param_interest):
    """ Similar to getShape, but with mass-splitting"""
    y = [[] for i in range(config.R_BIN+1)]
    for obj in idx_groups[group]:
        for rad in range(config.D_BINS+1):
            closest_idx = (np.abs(R - d[obj,rad]/d[obj,-int(config.D_LOGEND/((config.D_LOGEND-config.D_LOGSTART)/config.D_BINS))-1])).argmin() # Determine which point in R is closest
            if isnan(param_interest[obj][rad]) or np.log10(d[obj][rad]/d[obj,-int(config.D_LOGEND/((config.D_LOGEND-config.D_LOGSTART)/config.D_BINS))-1]) > config.R_LOGEND:
                continue
            else:
                y[closest_idx].append(param_interest[obj][rad])
    mean, err_low, err_high = getMeanOrMedianAndError(y)
    return mean, err_low, err_high

def labelLine(line,x,label=None,align=True,**kwargs):
    """ Used for IA spectra line labelling"""

    ax = line.axes
    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if (x < xdata[0]) or (x > xdata[-1]):
        print('x label location is outside data range!')
        return

    #Find corresponding y co-ordinate and angle of the line
    ip = 1
    for i in range(len(xdata)):
        if x < xdata[i]:
            ip = i
            break

    y = ydata[ip-1] + (ydata[ip]-ydata[ip-1])*(x-xdata[ip-1])/(xdata[ip]-xdata[ip-1])

    if not label:
        label = line.get_label()

    if align:
        #Compute the slope
        dx = xdata[ip] - xdata[ip-1]
        dy = ydata[ip] - ydata[ip-1]
        ang = degrees(atan2(dy,dx))

        #Transform to screen co-ordinates
        pt = np.array([x,y]).reshape((1,2))
        trans_angle = ax.transData.transform_angles(np.array((ang,)),pt)[0]

    else:
        trans_angle = 0

    #Set a bunch of keyword arguments
    if 'color' not in kwargs:
        kwargs['color'] = line.get_color()

    if ('horizontalalignment' not in kwargs) and ('ha' not in kwargs):
        kwargs['ha'] = 'center'

    if ('verticalalignment' not in kwargs) and ('va' not in kwargs):
        kwargs['va'] = 'center'

    if 'backgroundcolor' not in kwargs:
        kwargs['backgroundcolor'] = ax.get_facecolor()

    if 'clip_on' not in kwargs:
        kwargs['clip_on'] = True

    if 'zorder' not in kwargs:
        kwargs['zorder'] = 2.5
    
    ax.text(x,y,label,rotation=trans_angle,fontsize=8,**kwargs)

def labelLines(lines,align=True,xvals=None,**kwargs):
    """ Used for IA spectra line labelling"""
    
    ax = lines[0].axes
    labLines = []
    labels = []

    #Take only the lines which have labels other than the default ones
    for line in lines:
        label = line.get_label()
        if "_line" not in label:
            labLines.append(line)
            labels.append(label)

    if xvals is None:
        xmin,xmax = ax.get_xlim()
        xvals = np.linspace(xmin,xmax,len(labLines)+2)[1:-1]

    for line,x,label in zip(labLines,xvals,labels):
        labelLine(line,x,label,align,**kwargs)