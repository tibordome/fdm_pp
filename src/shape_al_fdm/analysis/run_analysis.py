#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 18:23:43 2021

@author: tibor
"""

from analyze_gx_alignments import analyze_gx_alignments
from analyze_gx_shapes import analyze_gx_shapes
from analyze_dm_alignments import analyze_dm_alignments
from analyze_dm_shapes import analyze_dm_shapes
from majors_projected_dm import projectMajorsSH
from majors_projected_gx import projectMajorsGx
import config
config.initialize()


# Gx alignment analysis
#analyze_gx_alignments()
#analyze_gx_shapes()

# SH alignment analysis
analyze_dm_alignments()
analyze_dm_shapes()

# Projection figures
#projectMajorsSH()
#projectMajorsGx()
