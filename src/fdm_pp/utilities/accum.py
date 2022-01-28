#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 14:24:59 2020

@author: tibor
"""

import numpy as np

def accumarray(subs, vals, size=None, fun=np.sum):

    if len(subs.shape) == 1:
        if size is None:
            size = [subs.values.max() + 1, 0]

        acc = vals.groupby(subs).agg(fun)
    else:
        if size is None:
            size = [subs.values.max()+1, subs.shape[1]] # not applicable

        subs = subs.copy().reset_index() # When we reset the index, the old index is added as a column, and a new sequential index is used
        by = subs.columns.tolist()[1:] # tolist() gives multi-dimensional list (i.e., list of lists), here it is [0,1]
        acc = subs.groupby(by=by)['index'].agg(list).apply(lambda x: vals[x].agg(fun)) # apply fun to entire list (1 list for given 0 (x) and 1 (y) column values)
        acc = acc.to_frame().reset_index().pivot_table(index=0, columns=1, aggfunc='first') # rearrange into x = row and y = column, 
        # Note that acc will be "padded" by NaNs (for those xy-entries for which list in previous line was empty)

    # Reindexing of Columns
    col_tmp = np.empty(0)
    for i in range(acc.columns.size):
        col_tmp = np.hstack((col_tmp, acc.columns[i][1]))
    acc.columns = col_tmp # for some reason columns were a MultiIndex object before, so turning into Float64Index
    acc = acc.reindex(range(size[1]), axis='columns').fillna(0) # cuts off last column (number N+1) if there is point with y = 0

    # Reindexing of Rows
    acc = acc.reindex(range(size[0])).fillna(0) # cuts off last row (number N+1) if there is point with x = 0

    return acc
