#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 21:09:19 2018

@author: mhyip
"""

import weak_con_func as wcf
import numpy as np
#from mpi4py import MPI

#import warnings
#warnings.filterwarnings("ignore", category=RuntimeWarning)
#%%
#comm = MPI.COMM_WORLD
#size=comm.Get_size()
#rank=comm.Get_rank()

dt_array= np.linspace(0.1,2,7)
#dt_array=np.array([1200, 1800, 2700, 3600, 5400, 7200, 10800])
dz=0.04
H=25.0
dtEu=0.1        
T=12*3600 
w=0


for i in range(len(dt_array)):
    print("\n Nest loop \n")
    
print("Done!")