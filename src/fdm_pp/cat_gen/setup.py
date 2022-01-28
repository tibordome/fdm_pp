#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 12:12:27 2020

@author: tibor
"""


from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

extension3 = [Extension(
                "fdm_utilities",
                sources=['fdm_utilities.pyx'],
                extra_compile_args=["-fopenmp"],
                extra_link_args=["-fopenmp"],
                include_dirs=[np.get_include(), '.']
            )]
    


setup(
    ext_modules = cythonize(extension3)
)

extension4 = [Extension(
                "make_grid_nn",
                sources=['make_grid_nn.pyx'],
                extra_compile_args=["-fopenmp"],
                extra_link_args=["-fopenmp"],
                include_dirs=[np.get_include(), '.']
            )]
    


setup(
    ext_modules = cythonize(extension4)
)