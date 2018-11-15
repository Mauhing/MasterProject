# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 16:41:25 2018

@author: mauhi
"""
#%% Include liberies 
# Import numpy and matplotlib, and use jupyter magic to
# get plots directly in notebook
import numpy as np
#import matplotlib
from matplotlib import pyplot as plt
# Get nicer looking plots than default
plt.style.use('bmh')
# Timer to measure the performance of methods
import functions as fs

from time import time
start=time()
#%% Initialize variable

w=0         #sink velocity
Tmax=1      #Maximum time
dt=1e-4     #Delta time
Ntime=int(Tmax/dt)  #Number of time interval
Np=4000             #Number of particles
z_e=np.zeros((Np,))  #initial the history table
z_e=np.random.uniform(0.7,0.70001,int(Np))     #Assign initial condition
z_m=z_e.copy()       #copy initialized history table to Milstein scheme
z_v=z_e.copy()       #copy initialized history table to Visser scheme
z_m2=z_e.copy()      #copy initialized history table to Milstein 2nd scheme

Nregi=10
Gap=int(Ntime/Nregi)

Gap=10          #Set to 1 if you want histogram to recorde all the timestep.
                #Or comment it, then histogram will skip some timestep.
Nbins=100       #Number of bin

start_time = time()

print("Ntime:  ", Ntime)

hist_E= np.zeros((1,Nbins-1))
hist_M= np.zeros((1,Nbins-1))
hist_V= np.zeros((1,Nbins-1))
hist_M2=np.zeros((1,Nbins-1))   


for i in range(Ntime-1):
    z_e=fs.step_e(z_e,w,dt,Np)

    z_m=fs.step_m(z_m,w,dt,Np)

    z_v=fs.step_v(z_v,w,dt,Np)

    z_m2=fs.step_m2(z_m2,w,dt,Np)

    if ( i % Gap ==0):
        temp, _ = np.histogram(z_e, bins = np.linspace(0, 1, Nbins))
        temp=temp.reshape(1,Nbins-1)
        hist_E=np.concatenate((hist_E,temp), axis=0)
        
        temp, _ = np.histogram(z_m, bins = np.linspace(0, 1, Nbins))
        temp=temp.reshape(1,Nbins-1)
        hist_M=np.concatenate((hist_M,temp), axis=0)

        temp, _ = np.histogram(z_v, bins = np.linspace(0, 1, Nbins))
        temp=temp.reshape(1,Nbins-1)
        hist_V=np.concatenate((hist_V,temp), axis=0)
    
        temp, _ = np.histogram(z_m2, bins = np.linspace(0, 1, Nbins))
        temp=temp.reshape(1,Nbins-1)
        hist_M2=np.concatenate((hist_M2,temp), axis=0)  
        
        print("\r ", i, end="\r",flush=True)
        #print(i, end=' ', flush=True)
        
        
#%%
bins = np.linspace(0, 1, Nbins)
midpoints = bins[:-1]+(bins[1]-bins[0])/2
times = np.linspace(0, Tmax, hist_E.shape[0])



for i in range(2):
    hist_E[i,:]=0
    hist_M[i,:]=0
    hist_V[i,:]=0
    hist_M2[i,:]=0


hist_E= hist_E.T
hist_M= hist_M.T
hist_V= hist_V.T
hist_M2=hist_M2.T

#plt.figure(1)
fig=plt.figure(1, figsize = (9,5))
plt.pcolormesh(times, midpoints, hist_E, cmap='jet')
plt.xlabel("Time")
plt.ylabel("Depth")
plt.tight_layout()
plt.grid(True)
fil="./figures/Euler_dt=%2.1e.png"%dt
#plt.savefig(fil)

#plt.figure(2)
fig=plt.figure(2, figsize = (9,5))
plt.pcolormesh(times, midpoints, hist_M, cmap='jet')
plt.xlabel("Time")
plt.ylabel("Depth")
plt.tight_layout()
plt.grid(True)
fil="./figures/Milstein_dt=%2.1e.png"%dt
#plt.savefig(fil)

#plt.figure(3)
fig=plt.figure(3, figsize = (9,5))
plt.pcolormesh(times, midpoints, hist_V, cmap='jet')
plt.xlabel("Time")
plt.ylabel("Depth")
plt.tight_layout()
plt.grid(True)
fil="./figures/Visse_dt=%2.1e.png"%dt
#plt.savefig(fil)

#plt.figure(4)
fig=plt.figure(4, figsize = (9,5))
plt.pcolormesh(times, midpoints, hist_M2, cmap='jet')
plt.xlabel("Time")
plt.ylabel("Depth")
plt.tight_layout()
plt.grid(True)
fil="./figures/Milstein2_dt=%2.1e.png"%dt
#plt.savefig(fil)


print("--- %s seconds ---" % (time() - start_time))

fig=plt.figure(5,figsize=(8,8))
plt.plot(np.mean(hist_E[:, -50:],axis=1), midpoints, label= "Euler")
plt.plot(np.mean(hist_V[:, -50:],axis=1), midpoints, label= "Visser")
plt.plot(np.mean(hist_M[:, -50:],axis=1), midpoints, label= "Milstein")
plt.plot(np.mean(hist_M2[:, -50:],axis=1), midpoints, label= "Milstein_2")
plt.show()

print("Time used: ", time()-start)