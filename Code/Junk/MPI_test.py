# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 14:48:22 2018

@author: mauhi
"""
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Create some np arrays on each process:
# For this demo, the arrays have only one
# entry that is assigned to be the rank of the processor
value = np.matrix([[2,5],[3,7]],'d')

print(' Rank: ',rank, ' value = ', value)
# initialize the np arrays that will store the results:
if rank==0:
    value_sum= np.matrix([[0,0],[0,0]],'d')

comm.barrier()

Sss=value.copy()
# perform the reductions:
comm.Reduce(value, value_sum, op=MPI.SUM, root=0)
if rank == 0:
    print(' Rank 0: value_sum = \n',Sss)

