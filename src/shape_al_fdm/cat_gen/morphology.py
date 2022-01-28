#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 13:17:51 2021

@author: tibor
"""

import numpy as np
import math
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
    xyz = respectPBC(xyz)
    com = np.sum(xyz*np.reshape(masses, (masses.shape[0],1)), axis = 0)/masses.sum() # COM of particle distribution
    xyz_new = xyz[(com[0]-xyz[:,0])**2+(com[1]-xyz[:,1])**2+(com[2]-xyz[:,2])**2 < d**2]-com
    masses_new = masses[(com[0]-xyz[:,0])**2+(com[1]-xyz[:,1])**2+(com[2]-xyz[:,2])**2 < d**2]
    xyz_new = xyz[(com[0]-xyz[:,0])**2+(com[1]-xyz[:,1])**2+(com[2]-xyz[:,2])**2 > (d-delta_d)**2]-com
    masses_new = masses[(com[0]-xyz[:,0])**2+(com[1]-xyz[:,1])**2+(com[2]-xyz[:,2])**2 > (d-delta_d)**2]
    err = 1; q_new = 1; s_new = 1; iteration = 1 # Start with spherical shell
    while (err > config.TOL):
        if iteration > config.N_WALL:
            #print("No convergence after {0} iterations for d being {1}".format(config.N_WALL, d))
            return np.nan, np.nan, np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan])
        if xyz_new.shape[0] < config.N_MIN:
            #print("Minimum number of particles has been undercut.")
            return np.nan, np.nan, np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan])
        shape_tensor = np.sum((masses_new)[:,np.newaxis,np.newaxis]*(np.matmul(xyz_new[:,:,np.newaxis],xyz_new[:,np.newaxis,:])),axis=0)/np.sum(masses_new)
        # Diagonalize shape_tensor
        try:
            eigval, eigvec = np.linalg.eigh(shape_tensor) # eigval is in ascending order, eigvec are orthogonal
        except np.linalg.LinAlgError:
            #print("Number of particles in current iteration is", xyz_new.shape[0], "hence LinAlgError for d being", d)
            return np.nan, np.nan, np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan])
        q = deepcopy(q_new); s = deepcopy(s_new)
        s_new, q_new = np.sqrt(eigval[:2]/eigval[2]) # In homoeoid approximation, eigenvalues are exactly a^2/3 etc...
        err = max(abs(q_new - q)/q, abs(s_new - s)/s) # Fractional differences
        rot_matrix = np.hstack((eigvec[2][:,np.newaxis]/np.linalg.norm(eigvec[2]), eigvec[1][:,np.newaxis]/np.linalg.norm(eigvec[1]), eigvec[0][:,np.newaxis]/np.linalg.norm(eigvec[0])))
        xyz_princ = np.zeros((xyz.shape[0],3))
        for i in range(xyz.shape[0]): # Transformation into the principal frame
            xyz_princ[i] = np.dot(rot_matrix, xyz[i]-com) 
        xyz_new = xyz[xyz_princ[:,0]**2+xyz_princ[:,1]**2/q_new**2+xyz_princ[:,2]**2/s_new**2 < d**2]-com
        masses_new = masses[xyz_princ[:,0]**2+xyz_princ[:,1]**2/q_new**2+xyz_princ[:,2]**2/s_new**2 < d**2]
        if q_new*d < delta_d or s_new*d < delta_d: # Condition true per definition for 1st shell. Also, can happen for 2nd: 2nd shell cannot cross 1st (though very rare).
            pass
        else:
            xyz_new = xyz[xyz_princ[:,0]**2/(d-delta_d)**2+xyz_princ[:,1]**2/(q_new*d-delta_d)**2+xyz_princ[:,2]**2/(s_new*d-delta_d)**2 > 1]-com
            masses_new = masses[xyz_princ[:,0]**2/(d-delta_d)**2+xyz_princ[:,1]**2/(q_new*d-delta_d)**2+xyz_princ[:,2]**2/(s_new*d-delta_d)**2 > 1]
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
    xyz: (N x 3) floats, position array
    masses: (N x 1) floats, mass array
    d: Distance from the COM, kept fixed during iterative procedure
    Returns:
    ------------
    q, s, v: Axis ratios evaluated at d, unit major axis vector
    """
    # COM respecting PBC
    xyz = respectPBC(xyz)
    com = np.sum(xyz*np.reshape(masses, (masses.shape[0],1)), axis = 0)/masses.sum() # COM of particle distribution
    xyz_new = xyz[(com[0]-xyz[:,0])**2+(com[1]-xyz[:,1])**2+(com[2]-xyz[:,2])**2 < d**2]-com
    masses_new = masses[(com[0]-xyz[:,0])**2+(com[1]-xyz[:,1])**2+(com[2]-xyz[:,2])**2 < d**2]
    err = 1; q_new = 1; s_new = 1; iteration = 1 # Start with sphere
    while (err > config.TOL):
        if iteration > config.N_WALL:
            #print("No convergence after {0} iterations for d being {1}".format(config.N_WALL, d))
            return np.nan, np.nan, np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan])
        if xyz_new.shape[0] < config.N_MIN:
            #print("Minimum number of particles has been undercut.")
            return np.nan, np.nan, np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan])
        shape_tensor = np.sum((masses_new)[:,np.newaxis,np.newaxis]*(np.matmul(xyz_new[:,:,np.newaxis],xyz_new[:,np.newaxis,:])),axis=0)/np.sum(masses_new)
        # Diagonalize shape_tensor
        try:
            eigval, eigvec = np.linalg.eigh(shape_tensor) # eigval is in ascending order, eigvec are orthogonal
        except np.linalg.LinAlgError:
            #print("Number of particles in current iteration is", xyz_new.shape[0], "hence LinAlgError for d being", d)
            return np.nan, np.nan, np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan])
        q = deepcopy(q_new); s = deepcopy(s_new)
        s_new, q_new = np.sqrt(eigval[:2]/eigval[2]) # It is assumed that eigenvalues are approximately proportional to a^2 etc., though there is no proof?!
        err = max(abs(q_new - q)/q, abs(s_new - s)/s) # Fractional differences
        rot_matrix = np.hstack((eigvec[2][:,np.newaxis]/np.linalg.norm(eigvec[2]), eigvec[1][:,np.newaxis]/np.linalg.norm(eigvec[1]), eigvec[0][:,np.newaxis]/np.linalg.norm(eigvec[0])))
        xyz_princ = np.zeros((xyz.shape[0],3))
        for i in range(xyz.shape[0]): # Transformation into the principal frame
            xyz_princ[i] = np.dot(rot_matrix, xyz[i]-com) 
        xyz_new = xyz[xyz_princ[:,0]**2+xyz_princ[:,1]**2/q_new**2+xyz_princ[:,2]**2/s_new**2 < d**2]-com
        masses_new = masses[xyz_princ[:,0]**2+xyz_princ[:,1]**2/q_new**2+xyz_princ[:,2]**2/s_new**2 < d**2]
        iteration += 1

    return q_new, s_new, eigvec[0]/np.linalg.norm(eigvec[0]), eigvec[1]/np.linalg.norm(eigvec[1]), eigvec[2]/np.linalg.norm(eigvec[2])

def getMorphology(xyz, cat, masses, obj_type, poolidx, start_time):
    """
    Calculates the axis ratios for the range [10**d_logstart, 10**d_logend] from the COM of the entire particle distro
    The ellipsoidal distance is always defined with respect to the COM of the entire particle distro, 
    not the COM of the initial spherical volume as in Katz 1991.
    Parameters:
    ------------
    xyz: (N1 x 3) floats, positions of particles
    cat: List of length N2, each entry a list containing indices of particles belonging to an object
    masses: (N1 x 1) floats, masses of the particles expressed in unit mass
    obj_type: string, either "DM" or "Gxs"
    poolidx: int, Index of object under investigation
    Returns:
    ------------
    l_q, l_s, l_major, l_inter, l_minor, l_coms: Axis ratios, major/intermediate/minor unit eigenvectors, COMs of objects
    Shape of, say, l_major: List of (number_of_objs) arrays, each array is of shape (d_discr, 3)
    """
    
    print_status(rank, start_time, "Dealing with object number {0}. The number of ptcs is {1}.".format(poolidx, len(cat[poolidx])), allowed_any = True)
    
    l_d = []
    l_q = []
    l_s = []
    l_major = []
    l_inter = []
    l_minor = []
    l_coms = []
    l_m = []
    obj = np.zeros((len(cat[poolidx]),3))
    masses_obj = np.zeros((len(cat[poolidx]),))
    for idx, ptc in enumerate(cat[poolidx]):
        obj[idx] = xyz[ptc]
        masses_obj[idx] = masses[ptc]
    obj = respectPBC(obj)
    l_coms.append(np.sum(obj*np.reshape(masses_obj, (masses_obj.shape[0],1)), axis = 0)/masses_obj.sum()) # COM of obj
    l_m.append(masses_obj.sum())
    
    d_max = np.max(np.array([np.linalg.norm(l_coms[-1]-obj[idx]) for idx in range(len(cat[poolidx]))]))
    d_max += config.SAFE
    
    d = np.logspace(config.D_LOGSTART,np.log10(d_max),config.D_BINS)
    if len(cat[poolidx]) > 10000: # If resolution is good, go for S1, as suggested by Zemp et al 2011
        qsv_obj = [S1_obj(obj, masses_obj, d[i], d[i]-d[i-1]) if i >= 1 else S1_obj(obj, masses_obj, d[i], d[i]) for i in range(d.shape[0])]
    else:
        qsv_obj = [E1_obj(obj, masses_obj, d[i]) for i in range(d.shape[0])]
    
    l_d.append(d)
    l_q.append(np.array(qsv_obj, dtype=object)[:,0])
    l_s.append(np.array(qsv_obj, dtype=object)[:,1])
    l_minor.append(np.array(qsv_obj, dtype=object)[:,2])
    l_inter.append(np.array(qsv_obj, dtype=object)[:,3])
    l_major.append(np.array(qsv_obj, dtype=object)[:,4])

    # Plotting
    if poolidx < 100:
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
        for idx in np.arange(l_major[-1].shape[0]-7, l_major[-1].shape[0]):
            if idx == l_major[-1].shape[0]-1:
                ax.quiver(*l_coms[-1], l_major[-1][idx][0], l_major[-1][idx][1], l_major[-1][idx][2], length=d[idx], color='m', label= "Major")
                ax.quiver(*l_coms[-1], l_inter[-1][idx][0], l_inter[-1][idx][1], l_inter[-1][idx][2], length=l_q[-1][idx]*d[idx], color='c', label = "Intermediate")
                ax.quiver(*l_coms[-1], l_minor[-1][idx][0], l_minor[-1][idx][1], l_minor[-1][idx][2], length=l_s[-1][idx]*d[idx], color='y', label = "Minor")
            else:
                ax.quiver(*l_coms[-1], l_major[-1][idx][0], l_major[-1][idx][1], l_major[-1][idx][2], length=d[idx], color='m')
                ax.quiver(*l_coms[-1], l_inter[-1][idx][0], l_inter[-1][idx][1], l_inter[-1][idx][2], length=l_q[-1][idx]*d[idx], color='c')
                ax.quiver(*l_coms[-1], l_minor[-1][idx][0], l_minor[-1][idx][1], l_minor[-1][idx][2], length=l_s[-1][idx]*d[idx], color='y')
        
        for special in np.arange(-7,-6):
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
        fontP = FontProperties()
        fontP.set_size('xx-small')
        plt.legend(bbox_to_anchor=(0.95, 1), loc='upper right', prop=fontP)        
        plt.xlabel(r"x (cMpc/h)")
        plt.ylabel(r"y (cMpc/h)")
        ax.set_zlabel(r"z (cMpc/h)")
        ax.set_box_aspect([1,1,1])
        set_axes_equal(ax)
        fig.savefig("{}/{}/{}{}.pdf".format(config.VIZ_DEST, obj_type, obj_type, poolidx+1), bbox_inches='tight')
    if math.isnan(l_q[-1][-1]): # Return empty lists if the outermost shell did not converge
        del l_d[-1]
        del l_q[-1]
        del l_s[-1]
        del l_minor[-1]
        del l_inter[-1]
        del l_major[-1]
        del l_coms[-1]
        del l_m[-1]
    return l_d, l_q, l_s, l_minor, l_inter, l_major, l_coms, l_m

def getMorphologies(xyz, cat, masses, obj_type, start_time):
    """
    Calculates the axis ratios for the range [10**d_logstart, 10**d_logend] from the COM of the entire particle distro
    The ellipsoidal distance is always defined with respect to the COM of the entire particle distro, 
    not the COM of the initial spherical volume as in Katz 1991.
    Parameters:
    ------------
    xyz: (N1 x 3) floats, positions of particles
    cat: List of length N2, each entry a list containing indices of particles belonging to an object
    masses: (N1 x 1) floats, masses of the particles expressed in unit mass
    obj_type: string, either "DM" or "Gxs"
    Returns:
    ------------
    l_q, l_s, l_major, l_inter, l_minor, l_coms: Axis ratios, major/intermediate/minor unit eigenvectors, COMs of objects
    Shape of, say, l_major: List of (number_of_objs) arrays, each array is of shape (d_discr, 3)
    """
    
    assert obj_type == "dm" or obj_type == "gxs"

    l_d = []
    l_q = []
    l_s = []
    l_major = []
    l_inter = []
    l_minor = []
    l_coms = []
    l_m = []

    perrank = len(cat)//size
    last = rank == size - 1 # Whether or not last process
    comm.Barrier()
    count = 0
    for obj in range(rank*perrank, (rank+1)*perrank+last*(len(cat)-(rank+1)*perrank)):
        l_d_obj, l_q_obj, l_s_obj, l_minor_obj, l_inter_obj, l_major_obj, l_coms_obj, l_m_obj = getMorphology(xyz, cat, masses, obj_type, obj, start_time)
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
    if rank == 0:
        l_d = [l_d[i][j] for i in range(size) for j in range(count_new[i])]
        l_q = [l_q[i][j] for i in range(size) for j in range(count_new[i])]
        l_s = [l_s[i][j] for i in range(size) for j in range(count_new[i])]
        l_minor = [l_minor[i][j] for i in range(size) for j in range(count_new[i])]
        l_inter = [l_inter[i][j] for i in range(size) for j in range(count_new[i])]
        l_major = [l_major[i][j] for i in range(size) for j in range(count_new[i])]
        l_coms = [l_coms[i][j] for i in range(size) for j in range(count_new[i])]
        l_m = [l_m[i][j] for i in range(size) for j in range(count_new[i])]
    return l_d, l_q, l_s, l_minor, l_inter, l_major, l_coms, l_m # Only rank = 0 content matters