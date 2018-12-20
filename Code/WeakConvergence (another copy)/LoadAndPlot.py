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

i = 3
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
plt.tight_layout()
plt.legend()
#plt.axis('equal')
#plt.savefig("figure")
#%%

momLagEM=np.load("Tor_momLagEM.npy")
momLagVi=np.load("Tor_momLagVi.npy")
momLagM1=np.load("Tor_momLagM1.npy")
momLagM2=np.load("Tor_momLagM2.npy")
crank_Eulerian=11.784612911560021
crank_Eulerian=11.784612911454307

#%%
#weak_conEM=np.load("Tor_Weak_EM.npy")
#weak_conM2=np.load("Tor_Weak_M2.npy")
#weak_conVi=np.load("Tor_Weak_Vi.npy")
#weak_conM1=np.load("Tor_Weak_M1.npy")
dtLa_array=np.load("Tor_dtArray.npy")


weak_conEM=np.abs(crank_Eulerian-momLagEM)
weak_conVi=np.abs(crank_Eulerian-momLagVi)
weak_conM1=np.abs(crank_Eulerian-momLagM1)
weak_conM2=np.abs(crank_Eulerian-momLagM2)






i = 2
fig=plt.figure(1)
l, = plt.plot(dtLa_array, weak_conEM,"-",label = "Euler_Ma")
plt.plot(dtLa_array, (weak_conEM[i]/dtLa_array[i]**1) * dtLa_array**1, "--",
         c = l.get_color(), label = "$\sim \Delta t^1$", alpha=0.3)

l, = plt.plot(dtLa_array, weak_conVi,"-",label = "Visser")
plt.plot(dtLa_array, (weak_conVi[i]/dtLa_array[i]**1) * dtLa_array**1, "--",
         c = l.get_color(), label = "$\sim \Delta t^1$", alpha=0.3)

l, = plt.plot(dtLa_array, weak_conM1,"-",label = "Milstein_1st")
plt.plot(dtLa_array, (weak_conM1[i]/dtLa_array[i]**1) * dtLa_array**1, "--",
         c = l.get_color(), label = "$\sim \Delta t^1$", alpha=0.3)

l, = plt.plot(dtLa_array, weak_conM2,"-",label = "Milstein_2nd")
plt.plot(dtLa_array, (weak_conM2[i]/dtLa_array[i]**2) * dtLa_array**2, "--", 
         c = l.get_color(), label = "$\sim \Delta t^2$", alpha=0.3)

plt.xlabel('dt (s)')
plt.ylabel('Absolute error')
plt.xscale('log')
plt.yscale('log')
plt.legend()
#plt.axis('equal')
#plt.savefig("figure")

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

