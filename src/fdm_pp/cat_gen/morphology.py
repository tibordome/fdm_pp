#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 13:17:51 2021

@author: tibor
"""

import numpy as np
from math import isnan
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib import pyplot
from print_msg import print_status
from mpl_toolkits.mplot3d import Axes3D
from fibonacci_sphere import fibonacci_ellipsoid
from set_axes_equal import set_axes_equal
import config
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

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

# S1 algorithm for subhalos/galaxies
def S1_obj(xyz, masses, d, delta_d):
    """
    Calculates the axis ratios at a distance d from the COM of the entire particle distro
    When keeping d fixed, it is always defined with respect to the COM of the entire particle distro, 
    not the COM of the initial spherical volume as in Katz 1991.
    Differential version of E1.
    Shells can cross (except 2nd shell with 1st), and a shell is assumed to be equally thick everywhere.
    Whether we adopt the last assumption or let the thickness float (Tomassetti et al 2016) barely makes 
    any difference in terms of shapes found, but the convergence properties improve for the version with fixated thickness.
    For 1st shell: delta_d is d
    Parameters:
    ------------
    xyz: (N x 3) floats, position array
    masses: (N x 1) floats, mass array
    d: Distance from the COM, kept fixed during iterative procedure
    delta_d: float, thickness of the shell in real space (constant across shells in logarithmic space)
    Returns:
    ------------
    q, s, eigframe: Axis ratios evaluated at d, unit major axis vectors ([0] gives minor, [1] intermediate, [2] major)
    """
    # PBC-respecting positions
    xyz = respectPBC(xyz)
    com = np.sum(xyz*np.reshape(masses, (masses.shape[0],1)), axis = 0)/masses.sum() # COM of particle distribution
    xyz_new = xyz[(com[0]-xyz[:,0])**2+(com[1]-xyz[:,1])**2+(com[2]-xyz[:,2])**2 < d**2]
    masses_new = masses[(com[0]-xyz[:,0])**2+(com[1]-xyz[:,1])**2+(com[2]-xyz[:,2])**2 < d**2]
    masses_new = masses_new[(com[0]-xyz_new[:,0])**2+(com[1]-xyz_new[:,1])**2+(com[2]-xyz_new[:,2])**2 >= (d-delta_d)**2]
    xyz_new = xyz_new[(com[0]-xyz_new[:,0])**2+(com[1]-xyz_new[:,1])**2+(com[2]-xyz_new[:,2])**2 >= (d-delta_d)**2]-com
    err = 1; q_new = 1; s_new = 1; iteration = 1 # Start with spherical shell
    while (err > config.TOL):
        if iteration > config.N_WALL:
            return np.nan, np.nan, np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan])
        if xyz_new.shape[0] < config.N_MIN:
            return np.nan, np.nan, np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan])
        shape_tensor = np.sum((masses_new)[:,np.newaxis,np.newaxis]*(np.matmul(xyz_new[:,:,np.newaxis],xyz_new[:,np.newaxis,:])),axis=0)/np.sum(masses_new)
        # Diagonalize shape_tensor
        try:
            eigval, eigvec = np.linalg.eigh(shape_tensor) # Note that eigval is in ascending order, and eigvec are orthogonal.
        except np.linalg.LinAlgError:
            return np.nan, np.nan, np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan])
        q = deepcopy(q_new); s = deepcopy(s_new)
        s_new, q_new = np.sqrt(eigval[:2]/eigval[2]) # It is assumed that eigenvalues are approximately proportional to a^2 etc. (true for uniform ellipsoid or uniform shell), though I have never seen any proof..
        err = max(abs(q_new - q)/q, abs(s_new - s)/s) # Fractional differences
        rot_matrix = np.hstack((eigvec[2][:,np.newaxis]/np.linalg.norm(eigvec[2]), eigvec[1][:,np.newaxis]/np.linalg.norm(eigvec[1]), eigvec[0][:,np.newaxis]/np.linalg.norm(eigvec[0])))
        xyz_princ = np.zeros((xyz.shape[0],3))
        for i in range(xyz.shape[0]):
            xyz_princ[i] = np.dot(rot_matrix, xyz[i]-com) 
        # Transformation into the principal frame
        if q_new*d <= delta_d or s_new*d <= delta_d: # Condition true per definition for 1st shell. But, can happen for 2nd: 2nd shell should not cross 1st (though very rare)!
            xyz_new = xyz[xyz_princ[:,0]**2+xyz_princ[:,1]**2/q_new**2+xyz_princ[:,2]**2/s_new**2 < d**2]-com
            masses_new = masses[xyz_princ[:,0]**2+xyz_princ[:,1]**2/q_new**2+xyz_princ[:,2]**2/s_new**2 < d**2]
        else:
            xyz_new = xyz[xyz_princ[:,0]**2+xyz_princ[:,1]**2/q_new**2+xyz_princ[:,2]**2/s_new**2 < d**2]
            xyz_princ_new = xyz_princ[xyz_princ[:,0]**2+xyz_princ[:,1]**2/q_new**2+xyz_princ[:,2]**2/s_new**2 < d**2]
            masses_new = masses[xyz_princ[:,0]**2+xyz_princ[:,1]**2/q_new**2+xyz_princ[:,2]**2/s_new**2 < d**2]
            masses_new = masses_new[xyz_princ_new[:,0]**2/(d-delta_d)**2+xyz_princ_new[:,1]**2/(q_new*d-delta_d)**2+xyz_princ_new[:,2]**2/(s_new*d-delta_d)**2 >= 1]
            xyz_new = xyz_new[xyz_princ_new[:,0]**2/(d-delta_d)**2+xyz_princ_new[:,1]**2/(q_new*d-delta_d)**2+xyz_princ_new[:,2]**2/(s_new*d-delta_d)**2 >= 1]-com
        iteration += 1
    return q_new, s_new, eigvec[0]/np.linalg.norm(eigvec[0]), eigvec[1]/np.linalg.norm(eigvec[1]), eigvec[2]/np.linalg.norm(eigvec[2])


# E1 algorithm for velocity dispersion, for subhalos/galaxies
def E1_vdisp(xyz, vxyz, masses, d):
    """
    Calculates the axis ratios at a distance d from the COM of the entire particle distro
    When keeping d fixed, it is always defined with respect to the COM of the entire particle distro, 
    not the COM of the initial spherical volume as in Katz 1991. One has to drop the < d condition
    in iteration number >= 2, since we diagonalize in velocity space.
    Parameters:
    ------------
    xyz: (N x 3) floats, positions of particles
    vxyz: (N x 3) floats, velocities of particles
    masses: (N x 1) floats, mass array
    d: Distance from the COM, for selecting particles at the beginning of the iteration
    Returns:
    ------------
    q, s, v: Axis ratios evaluated at d, unit major axis vector
    """
    
    # PBC-respecting positions
    xyz = respectPBC(xyz)
    com = np.sum(xyz*np.reshape(masses, (masses.shape[0],1)), axis = 0)/masses.sum() # COM of particle distribution
    vcom = np.sum(vxyz*np.reshape(masses, (masses.shape[0],1)), axis = 0)/masses.sum() # vCOM (velocity-COM) of particle distribution
    vxyz_new = vxyz[(com[0]-xyz[:,0])**2+(com[1]-xyz[:,1])**2+(com[2]-xyz[:,2])**2 < d**2]-vcom
    masses_new = masses[(com[0]-xyz[:,0])**2+(com[1]-xyz[:,1])**2+(com[2]-xyz[:,2])**2 < d**2]
    
    err = 1; q_new = 1; s_new = 1; iteration = 1 # Start with sphere
    while (err > config.TOL):
        if iteration > config.N_WALL:
            return np.nan, np.nan, np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan])
        vdisp_tensor = np.sum((masses_new)[:,np.newaxis,np.newaxis]*(np.matmul(vxyz_new[:,:,np.newaxis],vxyz_new[:,np.newaxis,:])),axis=0)/np.sum(masses_new)
        
        # Diagonalize velocity dispersion tensor
        try:
            eigval, eigvec = np.linalg.eigh(vdisp_tensor) # Note that eigval is in ascending order, and eigvec are orthogonal.
        except np.linalg.LinAlgError:
            return np.nan, np.nan, np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan])
        q = deepcopy(q_new); s = deepcopy(s_new)
        s_new, q_new = np.sqrt(eigval[:2]/eigval[2]) # It is assumed that eigenvalues are approximately proportional to a^2 etc. (true for uniform ellipsoid or uniform shell), though I have never seen any proof..
        err = max(abs(q_new - q)/q, abs(s_new - s)/s) # Fractional differences
        rot_matrix = np.hstack((eigvec[2][:,np.newaxis]/np.linalg.norm(eigvec[2]), eigvec[1][:,np.newaxis]/np.linalg.norm(eigvec[1]), eigvec[0][:,np.newaxis]/np.linalg.norm(eigvec[0])))
        xyz_princ = np.zeros((xyz.shape[0],3))
        for i in range(xyz.shape[0]): # Transformation into the principal frame
            xyz_princ[i] = np.dot(rot_matrix, xyz[i]-com) 
        vxyz_new = vxyz[xyz_princ[:,0]**2+xyz_princ[:,1]**2/q_new**2+xyz_princ[:,2]**2/s_new**2 < d**2]-vcom
        masses_new = masses[xyz_princ[:,0]**2+xyz_princ[:,1]**2/q_new**2+xyz_princ[:,2]**2/s_new**2 < d**2]
        iteration += 1
    
    return q_new, s_new, eigvec[0]/np.linalg.norm(eigvec[0]), eigvec[1]/np.linalg.norm(eigvec[1]), eigvec[2]/np.linalg.norm(eigvec[2])

# E1 algorithm for subhalos/galaxies
def E1_obj(xyz, masses, d):
    """
    Calculates the axis ratios at a distance d from the COM of the entire particle distro
    When keeping d fixed, it is always defined with respect to the COM of the entire particle distro, 
    not the COM of the initial spherical volume as in Katz 1991.
    Parameters:
    ------------
    xyz: (N x 3) floats, positions of particles
    masses: (N x 1) floats, mass array
    d: Distance from the COM, kept fixed during iterative procedure
    Returns:
    ------------
    q, s, v: Axis ratios evaluated at d, unit major axis vector
    """
    
    # PBC-respecting positions
    xyz = respectPBC(xyz)
    com = np.sum(xyz*np.reshape(masses, (masses.shape[0],1)), axis = 0)/masses.sum() # COM of particle distribution
    xyz_new = xyz[(com[0]-xyz[:,0])**2+(com[1]-xyz[:,1])**2+(com[2]-xyz[:,2])**2 < d**2]-com
    masses_new = masses[(com[0]-xyz[:,0])**2+(com[1]-xyz[:,1])**2+(com[2]-xyz[:,2])**2 < d**2]
    err = 1; q_new = 1; s_new = 1; iteration = 1 # Start with sphere
    while (err > config.TOL):
        if iteration > config.N_WALL:
            return np.nan, np.nan, np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan])
        if xyz_new.shape[0] < config.N_MIN:
            return np.nan, np.nan, np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan])
        shape_tensor = np.sum((masses_new)[:,np.newaxis,np.newaxis]*(np.matmul(xyz_new[:,:,np.newaxis],xyz_new[:,np.newaxis,:])),axis=0)/np.sum(masses_new)
        
        # Diagonalize shape_tensor
        try:
            eigval, eigvec = np.linalg.eigh(shape_tensor) # Note that eigval is in ascending order, and eigvec are orthogonal.
        except np.linalg.LinAlgError:
            return np.nan, np.nan, np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan])
        q = deepcopy(q_new); s = deepcopy(s_new)
        s_new, q_new = np.sqrt(eigval[:2]/eigval[2]) # It is assumed that eigenvalues are approximately proportional to a^2 etc. (true for uniform ellipsoid or uniform shell), though I have never seen any proof..
        err = max(abs(q_new - q)/q, abs(s_new - s)/s) # Fractional differences
        rot_matrix = np.hstack((eigvec[2][:,np.newaxis]/np.linalg.norm(eigvec[2]), eigvec[1][:,np.newaxis]/np.linalg.norm(eigvec[1]), eigvec[0][:,np.newaxis]/np.linalg.norm(eigvec[0])))
        xyz_princ = np.zeros((xyz.shape[0],3))
        for i in range(xyz.shape[0]): # Transformation into the principal frame
            xyz_princ[i] = np.dot(rot_matrix, xyz[i]-com) 
        xyz_new = xyz[xyz_princ[:,0]**2+xyz_princ[:,1]**2/q_new**2+xyz_princ[:,2]**2/s_new**2 < d**2]-com
        masses_new = masses[xyz_princ[:,0]**2+xyz_princ[:,1]**2/q_new**2+xyz_princ[:,2]**2/s_new**2 < d**2]
        iteration += 1

    return q_new, s_new, eigvec[0]/np.linalg.norm(eigvec[0]), eigvec[1]/np.linalg.norm(eigvec[1]), eigvec[2]/np.linalg.norm(eigvec[2])

def getMorphology(xyz, cat, masses, rdelta, obj_type, poolidx, purpose, start_time, vxyz = None):
    """
    Calculates the axis ratios for the range [10**d_logstart, 10**d_logend] from the COM of the entire particle distro
    The ellipsoidal distance is always defined with respect to the COM of the entire particle distro, 
    not the COM of the initial spherical volume as in Katz 1991.
    Parameters:
    ------------
    xyz: (N1 x 3) floats, positions of particles, or velocities (in case purpose is "vdisp")
    cat: List of length N2, each entry a list containing indices of particles belonging to an object
    masses: (N1 x 1) floats, masses of the particles expressed in unit mass
    rdelta: List of length N2, each entry giving the R_delta (mean not critical) radius of the parent halo
    obj_type: string, either "DM" or "Gxs"
    poolidx: int, Index of object under investigation
    purpose: string, either "local" or "overall", purpose of shape determination, either local shape is of interest, or just overall
    start_time: time.time() object, keeping track of time
    Returns:
    ------------
    l_q, l_s, l_major, l_inter, l_minor, l_coms: Axis ratios, major/intermediate/minor unit eigenvectors, COMs of objects
    Shape of, say, l_major: List of (number_of_objs) arrays, each array is of shape (d_discr, 3)
    """
    l_d = []
    l_q = []
    l_s = []
    l_major = []
    l_inter = []
    l_minor = []
    l_coms = []
    l_m = []
    d_max = rdelta[poolidx]
    
    if cat[poolidx] == []: # Too low resolution
        return l_d, l_q, l_s, l_minor, l_inter, l_major, l_coms, l_m
        
    obj = np.zeros((len(cat[poolidx]),3))
    masses_obj = np.zeros((len(cat[poolidx]),))
    for idx, ptc in enumerate(cat[poolidx]):
        obj[idx] = xyz[ptc]
        masses_obj[idx] = masses[ptc]
    obj = respectPBC(obj)
    l_coms.append(np.sum(obj*np.reshape(masses_obj, (masses_obj.shape[0],1)), axis = 0)/masses_obj.sum()) # COM of obj
    l_m.append(masses_obj.sum())
    
    if purpose == "local" and d_max == 0.0: # We are dealing with a halo which does not have any SHs, so R_delta = 0.0 according to AREPO
        l_d = []
        l_q = []
        l_s = []
        l_major = []
        l_inter = []
        l_minor = []
        l_coms = []
        l_m = []
        return l_d, l_q, l_s, l_minor, l_inter, l_major, l_coms, l_m
    if purpose == "local":
        d = d_max*np.logspace(config.D_LOGSTART,config.D_LOGEND,config.D_BINS+1)
    elif purpose == "overall":
        d = np.array([d_max+config.SAFE])
    else:
        d = np.array([d_max])
    if purpose == "local" or purpose == "overall":
        if len(cat[poolidx]) > 10000: # If resolution is good, go for S1, as suggested by Zemp et al 2011
            qsv_obj = [S1_obj(obj, masses_obj, d[i], d[i]-d[i-1]) if i >= 1 else S1_obj(obj, masses_obj, d[i], d[i]) for i in range(d.shape[0])]
        else:
            qsv_obj = [E1_obj(obj, masses_obj, d[i]) for i in range(d.shape[0])]
    else:
        vobj = np.zeros((len(cat[poolidx]),3))
        for idx, ptc in enumerate(cat[poolidx]):
            vobj[idx] = vxyz[ptc]
        qsv_obj = [E1_vdisp(obj, vobj, masses_obj, d[i]) for i in range(d.shape[0])]
    
    l_d.append(d)
    l_q.append(np.array(qsv_obj, dtype=object)[:,0])
    l_s.append(np.array(qsv_obj, dtype=object)[:,1])
    l_minor.append(np.array(qsv_obj, dtype=object)[:,2])
    l_inter.append(np.array(qsv_obj, dtype=object)[:,3])
    l_major.append(np.array(qsv_obj, dtype=object)[:,4])

    # Plotting
    if poolidx < 100 and config.SAVE_FIGS == True and (purpose == "local" or purpose == "overall"):
        fig = pyplot.figure()
        ax = Axes3D(fig, auto_add_to_figure = False)
        fig.add_axes(ax)
        ax.scatter(obj[:,0],obj[:,1],obj[:,2],s=masses_obj*2000, label = "Particles")
        ax.scatter(l_coms[-1][0],l_coms[-1][1],l_coms[-1][2],s=50,c="r", label = "COM")
        
        ell = fibonacci_ellipsoid(d[-1], l_q[-1][-1]*d[-1], l_s[-1][-1]*d[-1], samples=500)
        rot_matrix = np.hstack((np.reshape(l_major[-1][l_major[-1].shape[0]-1]/np.linalg.norm(l_major[-1][l_major[-1].shape[0]-1]), (3,1)), np.reshape(l_inter[-1][l_inter[-1].shape[0]-1]/np.linalg.norm(l_inter[-1][l_inter[-1].shape[0]-1]), (3,1)), np.reshape(l_minor[-1][l_minor[-1].shape[0]-1]/np.linalg.norm(l_minor[-1][l_minor[-1].shape[0]-1]), (3,1))))
        for j in range(len(ell)): # Transformation into the principal frame
            ell[j] = np.dot(rot_matrix, np.array(ell[j]))
        ell_x = np.array([x[0] for x in ell])
        ell_y = np.array([x[1] for x in ell])
        ell_z = np.array([x[2] for x in ell])
        ax.scatter(ell_x+l_coms[-1][0],ell_y+l_coms[-1][1],ell_z+l_coms[-1][2],s=1, c="g", label = "Inferred; a = {:.3}, b = {:.3}, c = {:.3}".format(d[-1], l_q[-1][-1]*d[-1], l_s[-1][-1]*d[-1]))
        if purpose == "local":
            for idx in np.arange(l_major[-1].shape[0]-config.D_BINS//5, l_major[-1].shape[0]):
                if idx == l_major[-1].shape[0]-1:
                    ax.quiver(*l_coms[-1], l_major[-1][idx][0], l_major[-1][idx][1], l_major[-1][idx][2], length=d[idx], color='m', label= "Major")
                    ax.quiver(*l_coms[-1], l_inter[-1][idx][0], l_inter[-1][idx][1], l_inter[-1][idx][2], length=l_q[-1][idx]*d[idx], color='c', label = "Intermediate")
                    ax.quiver(*l_coms[-1], l_minor[-1][idx][0], l_minor[-1][idx][1], l_minor[-1][idx][2], length=l_s[-1][idx]*d[idx], color='y', label = "Minor")
                else:
                    ax.quiver(*l_coms[-1], l_major[-1][idx][0], l_major[-1][idx][1], l_major[-1][idx][2], length=d[idx], color='m')
                    ax.quiver(*l_coms[-1], l_inter[-1][idx][0], l_inter[-1][idx][1], l_inter[-1][idx][2], length=l_q[-1][idx]*d[idx], color='c')
                    ax.quiver(*l_coms[-1], l_minor[-1][idx][0], l_minor[-1][idx][1], l_minor[-1][idx][2], length=l_s[-1][idx]*d[idx], color='y')
            for special in np.arange(-config.D_BINS//5,-config.D_BINS//5+1):
                ell = fibonacci_ellipsoid(d[special], l_q[-1][special]*d[special], l_s[-1][special]*d[special], samples=500)
                rot_matrix = np.hstack((np.reshape(l_major[-1][special]/np.linalg.norm(l_major[-1][special]), (3,1)), np.reshape(l_inter[-1][special]/np.linalg.norm(l_inter[-1][special]), (3,1)), np.reshape(l_minor[-1][special]/np.linalg.norm(l_minor[-1][special]), (3,1))))
                for j in range(len(ell)): # Transformation into the principal frame
                    ell[j] = np.dot(rot_matrix, np.array(ell[j])) 
                ell_x = np.array([x[0] for x in ell])
                ell_y = np.array([x[1] for x in ell])
                ell_z = np.array([x[2] for x in ell])
                ax.scatter(ell_x+l_coms[-1][0],ell_y+l_coms[-1][1],ell_z+l_coms[-1][2],s=1, c="r", label = "Inferred; a = {:.3}, b = {:.3}, c = {:.3}".format(d[special], l_q[-1][special]*d[special], l_s[-1][special]*d[special]))
                ax.quiver(*l_coms[-1], l_major[-1][special][0], l_major[-1][special][1], l_major[-1][special][2], length=d[special], color='limegreen', label= "Major {0}".format(special))
                ax.quiver(*l_coms[-1], l_inter[-1][special][0], l_inter[-1][special][1], l_inter[-1][special][2], length=l_q[-1][special]*d[special], color='darkorange', label = "Intermediate {0}".format(special))
                ax.quiver(*l_coms[-1], l_minor[-1][special][0], l_minor[-1][special][1], l_minor[-1][special][2], length=l_s[-1][special]*d[special], color='indigo', label = "Minor {0}".format(special))
        else:
            ax.quiver(*l_coms[-1], l_major[-1][-1][0], l_major[-1][-1][1], l_major[-1][-1][2], length=d[-1], color='m', label= "Major")
            ax.quiver(*l_coms[-1], l_inter[-1][-1][0], l_inter[-1][-1][1], l_inter[-1][-1][2], length=l_q[-1][-1]*d[-1], color='c', label = "Intermediate")
            ax.quiver(*l_coms[-1], l_minor[-1][-1][0], l_minor[-1][-1][1], l_minor[-1][-1][2], length=l_s[-1][-1]*d[-1], color='y', label = "Minor")
            
        fontP = FontProperties()
        fontP.set_size('xx-small')
        plt.legend(bbox_to_anchor=(0.95, 1), loc='upper right', prop=fontP)        
        plt.xlabel(r"x (cMpc/h)")
        plt.ylabel(r"y (cMpc/h)")
        ax.set_zlabel(r"z (cMpc/h)")
        ax.set_box_aspect([1,1,1])
        set_axes_equal(ax)
        fig.savefig("{}/{}/{}/{}{}.pdf".format(config.VIZ_DEST, config.DM_TYPE, obj_type, obj_type, poolidx+1), bbox_inches='tight')
    if purpose == "local" and isnan(l_q[-1][np.argmin(abs(d - d_max))]): # Return empty lists if rdelta shell did not converge
        l_d = []
        l_q = []
        l_s = []
        l_major = []
        l_inter = []
        l_minor = []
        l_coms = []
        l_m = []
    return l_d, l_q, l_s, l_minor, l_inter, l_major, l_coms, l_m
    
def getMorphologies(xyz, cat, masses, rdelta, obj_type, purpose, start_time, vxyz = None):
    """
    Calculates the axis ratios for the range [10**d_logstart, 10**d_logend] from the COM of the entire particle distro
    The ellipsoidal distance is always defined with respect to the COM of the entire particle distro, 
    not the COM of the initial spherical volume as in Katz 1991.
    Parameters:
    ------------
    xyz: (N1 x 3) floats, positions of particles
    vxyz: (N1 x 3) floats, velocities (in case purpose is "vdisp")
    cat: List of length N2, each entry a list containing indices of particles belonging to an object
    masses: (N1 x 1) floats, masses of the particles expressed in unit mass
    rdelta: List of length N2, each entry giving the R_delta (mean not critical) radius of the parent halo
    obj_type: string, either "dm" or "gxs"
    purpose: string, either "local" or "overall" or "vdisp", purpose of shape determination, either local shape is of interest, or just overall
             or velocity dispersion determination
    start_time: time.time() object, keeping track of time
    Returns:
    ------------
    if purpose == "overall":
        l_d, l_q, l_s, l_major, l_inter, l_minor, l_coms: Axis ratios, major/intermediate/minor unit eigenvectors, COMs of objects
    elif purpose == "local":
        l_d, l_q, l_s, l_major, l_inter, l_minor, l_coms, l_succeeded: Axis ratios, major/intermediate/minor unit eigenvectors, 
                                                                  COMs of objects, which objects' shape calculation succeeded
    else: # purpose == "vdisp"
        l_major
    Shape of, say, l_major: List of (number_of_objs) arrays, each array is of shape (d_discr, 3)
    """
    
    assert obj_type == "dm" or obj_type == "gxs"
    assert purpose == "overall" or purpose == "local" or purpose == "vdisp"

    l_d = []
    l_q = []
    l_s = []
    l_major = []
    l_inter = []
    l_minor = []
    l_coms = []
    l_m = []
    l_succeeded = []

    nb_jobs_to_do = len(cat)
    perrank = nb_jobs_to_do//size + (nb_jobs_to_do//size == 0)*1
    do_sth = rank <= nb_jobs_to_do-1
    if size <= nb_jobs_to_do:
        last = rank == size - 1 # Whether or not last process
    else:
        last = rank == nb_jobs_to_do - 1
    comm.Barrier()
    count = 0
    for obj in range(rank*perrank, rank*perrank+do_sth*(perrank+last*(nb_jobs_to_do-(rank+1)*perrank))):
        l_d_obj, l_q_obj, l_s_obj, l_minor_obj, l_inter_obj, l_major_obj, l_coms_obj, l_m_obj = getMorphology(xyz, cat, masses, rdelta, obj_type, obj, purpose, start_time, vxyz)
        if len(cat[obj]) != 0:
            if purpose == "overall":
                print_status(rank, start_time, "Purpose: {0}. Dealing with object number {1}. The number of ptcs is {2}. Shape determination at outermost ell. radius successful: {success}".format(purpose, obj, len(cat[obj]), success="False" if l_d_obj == [] or isnan(l_major_obj[0][-1][0]) else "True"), allowed_any = True)
            elif purpose == "local":
                print_status(rank, start_time, "Purpose: {0}. Dealing with object number {1}. The number of ptcs is {2}. Shape determination at outermost ell. radius or rdelta successful: {success}".format(purpose, obj, len(cat[obj]), success="False" if l_d_obj == [] or isnan(l_major_obj[0][-1][0]) else "True"), allowed_any = True)
            else:
                print_status(rank, start_time, "Purpose: {0}. Dealing with object number {1}. The number of ptcs is {2}. Shape determination at rdelta successful: {success}".format(purpose, obj, len(cat[obj]), success="False" if l_d_obj == [] or isnan(l_major_obj[0][-1][0]) else "True"), allowed_any = True)
        l_d += l_d_obj
        l_q += l_q_obj
        l_s += l_s_obj
        l_minor += l_minor_obj
        l_inter += l_inter_obj
        l_major += l_major_obj
        l_coms += l_coms_obj
        l_m += l_m_obj
        if l_d_obj != []:
            count += 1
            l_succeeded += [obj]
    comm.Barrier()
    count_new = comm.gather(count, root=0)
    l_d = comm.gather(l_d, root = 0)
    l_q = comm.gather(l_q, root = 0)
    l_s = comm.gather(l_s, root = 0)
    l_minor = comm.gather(l_minor, root = 0)
    l_inter = comm.gather(l_inter, root = 0)
    l_major = comm.gather(l_major, root = 0)
    l_coms = comm.gather(l_coms, root = 0)
    l_m = comm.gather(l_m, root = 0)
    l_succeeded = comm.gather(l_succeeded, root = 0)
    if rank == 0:
        l_d = [l_d[i][j] for i in range(size) for j in range(count_new[i])]
        l_q = [l_q[i][j] for i in range(size) for j in range(count_new[i])]
        l_s = [l_s[i][j] for i in range(size) for j in range(count_new[i])]
        l_minor = [l_minor[i][j] for i in range(size) for j in range(count_new[i])]
        l_inter = [l_inter[i][j] for i in range(size) for j in range(count_new[i])]
        l_major = [l_major[i][j] for i in range(size) for j in range(count_new[i])]
        l_coms = [l_coms[i][j] for i in range(size) for j in range(count_new[i])]
        l_m = [l_m[i][j] for i in range(size) for j in range(count_new[i])]
        l_succeeded = [l_succeeded[i][j] for i in range(size) for j in range(count_new[i])]
    if purpose == "local":
        return l_d, l_q, l_s, l_minor, l_inter, l_major, l_coms, l_m, l_succeeded # Only rank = 0 content matters
    elif purpose == "overall":
        return l_d, l_q, l_s, l_minor, l_inter, l_major, l_coms, l_m # Only rank = 0 content matters
    else:
        purpose == "vdisp"
        return l_q, l_s, l_major