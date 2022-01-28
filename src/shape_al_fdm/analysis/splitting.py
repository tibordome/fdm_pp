#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 15:42:52 2021

@author: tibor
"""

import config
import numpy as np

def chunks(lst, n):
    #Yield successive n-sized chunks from lst (list or np.array).
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
            
def R_split(seps, split_type = "just_r_smaller_2"):
    max_min_r = []
    number_of_groups = 0
    if split_type == "just_r_smaller_2":
        max_min_r.append(np.array(seps)[:,2].min())
        max_min_r.append(2.0)
        number_of_groups = 1
    else:
        assert split_type == "proper_split"
        delta_r = (np.array(seps)[:,2].max() - np.array(seps)[:,2].min())/(config.R_SLICES+1)
        for i in range(config.R_SLICES + 2):
            if i == config.R_SLICES+1:
                max_min_r.append(np.array(seps)[:,2].max())
            else:
                max_min_r.append(np.array(seps)[:,2].min()+delta_r*i)
        number_of_groups = config.R_SLICES+1
    return max_min_r, number_of_groups

def M_split(m, com, v = None):
    """M splitting
    Arguments:
    -----------
    m: list of floats, DM masses of Gxs
    com: list of (3,) float arrays, COMs of Gxs
    config.M_SPLIT_TYPE: either "log_slice", where Gxs masses in log space are split, 
    or "const_occ", where masses are split ensuring equal number of points in each bin
    v: list of (3,) float arrays, major axis of Gxs (optional)
    Return:
    ------------
    max_min_m, m_groups, gx_com_groups, v_groups, idx_groups
    """
    
    args_sort = np.argsort(m)
    m_ordered = np.sort(m)
    max_min_m = []
    max_min_mlog = []
    m_groups = []
    gx_com_groups = []
    v_groups = []
    if len(m) % config.M_BINS_C == 0:
        chunk_size = len(m)//config.M_BINS_C
    else:
        chunk_size = len(m)//config.M_BINS_C + 1
    if config.M_SPLIT_TYPE == "const_occ":
        m_groups = list(chunks(list(m_ordered), chunk_size)) # List of lists
        gx_com_groups = list(chunks(com[args_sort], chunk_size)) # List of arrays
        if v is not None:
            v_groups = list(chunks(v[args_sort], chunk_size)) # List of arrays
        print("The groups (except maybe last) have size", chunk_size)
        print("The group number is", len(m_groups))
        for i in range(len(m_groups)+1):
            if i == len(m_groups):
                print("Masses are the following", m_ordered[-1])
                max_min_m.append(m_ordered[-1])
            else:
                print("Masses are the following:", m_ordered[i*chunk_size])
                max_min_m.append(m_ordered[i*chunk_size])
    else:
        assert config.M_SPLIT_TYPE == "log_slice"
        log_m_min = np.log10(m.min())
        log_m_max = np.log10(m.max())
        delta_m = (log_m_max - log_m_min)/(config.M_SLICES+1)
        for i in range(config.M_SLICES + 2):
            if i == config.M_SLICES+1:
                max_min_m.append(10**(log_m_max))
                max_min_mlog.append(log_m_max)
            else:
                max_min_m.append(10**(log_m_min+delta_m*i))
                max_min_mlog.append(log_m_min+delta_m*i)
        split_occ = np.zeros((config.M_SLICES+1,))
        for m in range(len(m_ordered)):
            for split in range(config.M_SLICES+1):
                if np.log10(m_ordered[m]) >= max_min_mlog[split] and np.log10(m_ordered[m]) <= max_min_mlog[split+1]:
                    split_occ[split] += 1
        for split in range(config.M_SLICES+1):
            m_groups.append([m_ordered[i] for i in np.arange(int(np.array([split_occ[j] for j in range(split)]).sum()), int(np.array([split_occ[j] for j in range(split+1)]).sum()))])
            gx_com_groups.append(np.array([com[args_sort][i] for i in np.arange(int(np.array([split_occ[j] for j in range(split)]).sum()), int(np.array([split_occ[j] for j in range(split+1)]).sum()))]))
            if v is not None:
                v_groups.append(np.array([v[args_sort][i] for i in np.arange(int(np.array([split_occ[j] for j in range(split)]).sum()), int(np.array([split_occ[j] for j in range(split+1)]).sum()))]))
        print("Split occupancies:", split_occ)
    idx_groups = [[args_sort[i] for i in np.arange(int(np.array([len(m_groups[k]) for k in range(j)]).sum()), int(np.array([len(m_groups[k]) for k in range(j)]).sum())+len(m_groups[j]))] for j in range(len(m_groups))]
    return max_min_m, m_groups, gx_com_groups, v_groups, idx_groups
