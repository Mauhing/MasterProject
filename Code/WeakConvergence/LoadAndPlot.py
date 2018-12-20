#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 12:06:13 2018

@author: mhyip
"""


from matplotlib import pyplot as plt
plt.style.use('bmh')
import numpy as np

#%%
weak_conEM=np.load("Weak_EM.npy")
weak_conM2=np.load("Weak_M2.npy")
weak_conVi=np.load("Weak_Vi.npy")
weak_conM1=np.load("Weak_M1.npy")
dtLa_array=np.load("dtArray.npy")

i = 2
fig=plt.figure(1)
l, = plt.plot(dtLa_array, weak_conEM,".",label = "Euler_Ma")
plt.plot(dtLa_array, (weak_conEM[i]/dtLa_array[i]**1) * dtLa_array**1, "--",
         c = l.get_color(), label = "$\sim \Delta t^1$", alpha=0.3)

l, = plt.plot(dtLa_array, weak_conVi,".",label = "Visser")
plt.plot(dtLa_array, (weak_conVi[i]/dtLa_array[i]**1) * dtLa_array**1, "--",
         c = l.get_color(), label = "$\sim \Delta t^1$", alpha=0.3)

l, = plt.plot(dtLa_array, weak_conM1,".",label = "Milstein_1st")
plt.plot(dtLa_array, (weak_conM1[i]/dtLa_array[i]**1) * dtLa_array**1, "--",
         c = l.get_color(), label = "$\sim \Delta t^1$", alpha=0.3)

l, = plt.plot(dtLa_array, weak_conM2,"*",label = "Milstein_2nd")
plt.plot(dtLa_array, (weak_conM2[i]/dtLa_array[i]**2) * dtLa_array**2, "--", 
         c = l.get_color(), label = "$\sim \Delta t^2$", alpha=0.3)

plt.xlabel('dt (s)')
plt.ylabel('Absolute error')
plt.xscale('log')
plt.yscale('log')
plt.legend()
#plt.axis('equal')
plt.savefig("figure")

#%%
"""
zEu=np.load("zEu.npy")
Conc=np.load("Concentration.npy")
midpoints=np.load("midpointsWeakConv.npy")
hist=np.load("histOfWeakConv.npy")

fig=plt.figure(2)

plt.plot(zEu,Conc, label="Eulerian")   
plt.plot(midpoints,hist,".",alpha=0.4, label="Lagrangian Milstein 2nd")
plt.xlabel('Depth (m)')
plt.ylabel('Concentration')
plt.legend()
"""

