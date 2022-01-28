#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 18:26:34 2021

@author: tibor
"""

import time

# Routine to print script status to command line, with elapsed time
def print_status(rank,start_time,message,allowed_any=False):
    if rank == 0 or allowed_any == True:
        elapsed_time = time.time() - start_time
        print('%d\ts: %s' % (elapsed_time,message))