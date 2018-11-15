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
#matplotlib inline
from matplotlib import pyplot as plt
# Get nicer looking plots than default
plt.style.use('bmh')
# Timer to measure the performance of methods
from time import time
# Maybe useful. But I am using it. 

#%%
#Calculate nest step


#######
#Euler#
#######
def step_e(z,w,K,dt,N_sample):

    dW=np.random.normal(0,np.sqrt(dt),N_sample)
    return z+alpha(w,z)*dt+beta(z)*dW

##########
#Milstein#
##########
def step_m(z,w,K,dt,N_sample):
    dW=np.random.normal(0,np.sqrt(dt),N_sample)
    return z + w*dt + (1/2)*dKdz(z)*(dW*dW+dt) + beta(z)*dW

########
#Visser#
########
def step_m2(z,w,K,dt,N_sample):
    dW=np.random.normal(0,np.sqrt(dt),N_sample)
    return z + alpha(w,z)*dt+beta(z)*dW+\
            1/2*beta(z)*dBdz(z)*(dW*dW-dt)+\
            1/2*(dABdz(w,z)+1/2*ddBdzz(z)*beta(z)**2)*dW*dt+\
            1/2*(alpha(w,z)*dAdz(w,z)+1/2*ddAdzz(w,z)*beta(z)**2)*dt**2

##############
#Milstein 2nd#
##############
def step_v(z,w,K,dt,N_sample):
    dW=np.random.normal(0,np.sqrt(dt),N_sample)
    return z + alpha(w,z)*dt + beta(z + 1/2*alpha(w,z)*dt)*dW

#%%
#Some ultility function
def alpha(w,z):
    return w+dKdz(z)

def dAdz(w,z):
    return  (alpha(w,z+dz)-alpha(w,z-dz))/(2*dz)

def ddAdzz(w,z):
    return (dAdz(w,z+dz)-dAdz(w,z-dz))/(2*dz)

def beta(z):
    return np.sqrt(2*K(z))

def dBdz(z):
    return (beta(z+dz)-beta(z-dz))/(2*dz)

def ddBdzz(z):
    return (dBdz(z+dz)-dBdz(z-dz))/(2*dz)

def dABdz(w,z):
    return (alpha(w,z+dz)*beta(z+dz)-alpha(w,z-dz)*beta(z-dz))/(2*dz)



# Pycnocline
def pycnocline(z):
    # See Gräwe et al. (2012)
    a = 1
    H = 1
    Kbar = 1
    prefactor = Kbar*2*(1+a)*(1+2*a)/(a**2*H**(1+1/a))
    G = prefactor*np.where(z < H/2, z*(H-2*z)**(1/a),
                           (H-z)*(2*z-1)**(1/a))
    return np.where((0.0 < z) & (z < 1.0), G, 0.0)

def dKdz(z): #Derivative of pycnocline
    ##temp=(pycnocline(z+dz)-pycnocline(z-dz))/(2*dz)
    a = 1
    H = 1
    Kbar = 1
    prefactor = Kbar*2*(1+a)*(1+2*a)/(a**2*H**(1+1/a))
    
    return prefactor*np.where(z < H/2, (H - 2*z)**(1/a) - 2*z*(H - 2*z)**(1/a)/(a*(H - 2*z))
                              ,-(2*z - 1)**(1/a) + 2*(H - z)*(2*z - 1)**(1/a)/(a*(2*z - 1)))

#%% Globel variabl
dz=1e-6

#%% Initialize variable
w=0         #sink velocity
Tmax=1    #Maximum time
dt=1e-3     #Delta time
Ntime=int(Tmax/dt)  #Number of time interval
Np=4000          #Number of particles
K=pycnocline        #Define K to pycnocline function
z_e=np.zeros((Np,))  #initial the history table
z_e=np.random.uniform(0.7,0.70001,int(Np))     #Assign initial condition
z_m=z_e.copy()       #copy initialized history table to Milstein scheme
z_v=z_e.copy()       #copy initialized history table to Visser scheme
z_m2=z_e.copy()      #copy initialized history table to Milstein 2nd scheme

Nregi=10
Gap=int(Ntime/Nregi)

Gap=10           #Set to 1 if you want histogram to recorde all the timestep.
                #Or comment it, then histogram will skip some timestep.
Nbins=100           #Number of bin

start_time = time()

print("Ntime:  ", Ntime)

hist_E= np.zeros((1,Nbins-1))
hist_M= np.zeros((1,Nbins-1))
hist_V= np.zeros((1,Nbins-1))
hist_M2=np.zeros((1,Nbins-1))   


for i in range(Ntime-1):
    z_e=step_e(z_e,w,K,dt,Np)
    z_e=np.where(z_e>1.0, 1-(z_e-1),z_e)
    z_e=np.where(z_e<0.0, -z_e,z_e)
    
    z_m=step_m(z_m,w,K,dt,Np)
    z_m=np.where(z_m>1.0, 1-(z_m-1),z_m)
    z_m=np.where(z_m<0.0, -z_m,z_m)
    
    z_v=step_v(z_v,w,K,dt,Np)
    z_v=np.where(z_v>1.0, 1-(z_v-1),z_v)
    z_v=np.where(z_v<0.0, -z_v,z_v)
    
    z_m2=step_m2(z_m2,w,K,dt,Np)
    z_m2=np.where(z_m2>1.0, 1-(z_m2-1),z_m2)
    z_m2=np.where(z_m2<0.0, -z_m2,z_m2)
    
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
        
        print(i, end=' ', flush=True)
        
        
        
        
        
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
fil="Euler_dt=%2.1e.png"%dt
plt.savefig(fil)

#plt.figure(2)
fig=plt.figure(2, figsize = (9,5))
plt.pcolormesh(times, midpoints, hist_V, cmap='jet')
plt.xlabel("Time")
plt.ylabel("Depth")
plt.tight_layout()
plt.grid(True)
fil="Visse_dt=%2.1e.png"%dt
plt.savefig(fil)

#plt.figure(3)
fig=plt.figure(3, figsize = (9,5))
plt.pcolormesh(times, midpoints, hist_M, cmap='jet')
plt.xlabel("Time")
plt.ylabel("Depth")
plt.tight_layout()
plt.grid(True)
fil="Milstein_dt=%2.1e.png"%dt
plt.savefig(fil)

#plt.figure(4)
fig=plt.figure(4, figsize = (9,5))
plt.pcolormesh(times, midpoints, hist_M2, cmap='jet')
plt.xlabel("Time")
plt.ylabel("Depth")
plt.tight_layout()
plt.grid(True)
fil="Milstein2_dt=%2.1e.png"%dt
plt.savefig(fil)

plt.show()
print("--- %s seconds ---" % (time() - start_time))

fig=plt.figure(figsize=(8,8))
plt.plot(np.mean(hist_E[:, -50:],axis=1), midpoints, label= "Euler")
plt.plot(np.mean(hist_V[:, -50:],axis=1), midpoints, label= "Visser")
plt.plot(np.mean(hist_M[:, -50:],axis=1), midpoints, label= "Milstein")
plt.plot(np.mean(hist_M2[:, -50:],axis=1), midpoints, label= "Milstein_2")



































