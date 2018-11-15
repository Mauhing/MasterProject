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
from scipy.misc import derivative

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
    a = 1
    H = 1
    Kbar = 1
    prefactor = Kbar*2*(1+a)*(1+2*a)/(a**2*H**(1+1/a))
    return prefactor*np.where(z < H/2
                              , 4*(H - 2*z)**(1/a)*(-z/(H - 2*z) - 1\
                                   + z/(a*(H - 2*z)))/(a*(H - 2*z))
                              ,4*(2*z - 1)**(1/a)*(-(H - z)/(2*z - 1) - 1\
                                  + (H - z)/(a*(2*z - 1)))/(a*(2*z - 1)))

def beta(z):
    return np.sqrt(2*K(z))

def dABdz(w,z):
    return (alpha(w,z+dz)*beta(z+dz)-alpha(w,z-dz)*beta(z-dz))/(2*dz)
    

def dBdz(z):
    a = 1
    H = 1
    Kbar = 1
    prefactor = np.sqrt(2*Kbar*2*(1+a)*(1+2*a)/(a**2*H**(1+1/a))) 
    
    for x in np.nditer(z):
        if x<0 or x>1:
            print("something wrong: ", x)
            
    return prefactor*np.where(z < H/2
                              , np.sqrt(z*(H - 2*z)**(1/a))*(H - 2*z)**(-1/a)*((H - 2*z)**(1/a)/2\
                                        - z*(H - 2*z)**(1/a)/(a*(H - 2*z)))/z
                              , np.sqrt((H - z)*(2*z - 1)**(1/a))*(2*z - 1)**(-1/a)*(-(2*z - 1)**(1/a)/2\
                                        + (H - z)*(2*z - 1)**(1/a)/(a*(2*z - 1)))/(H - z))
    
def ddBdzz(z):
    a = 1
    H = 1
    Kbar = 1
    prefactor = np.sqrt(2*Kbar*2*(1+a)*(1+2*a)/(a**2*H**(1+1/a)))
    
    for x in np.nditer(z):
        if x<0 or x>1:
            print("something wrong: ", x)
            
    return prefactor*np.where(z < H/2
                              , np.sqrt(z*(H - 2*z)**(1/a))*((1 - 2*z/(a*(H - 2*z)))**2/(4*z)\
                                        - (1 - 2*z/(a*(H - 2*z)))/(2*z)\
                                        + (1 - 2*z/(a*(H - 2*z)))/(a*(H - 2*z))\
                                        - 2*(z/(H - 2*z)\
                                        + 1 - z/(a*(H - 2*z)))/(a*(H - 2*z)))/z
                              , np.sqrt((H - z)*(2*z - 1)**(1/a))*((1 - 2*(H - z)/(a*(2*z - 1)))**2/(4*(H - z))\
                                        - (1 - 2*(H - z)/(a*(2*z - 1)))/(2*(H - z))\
                                        + (1 - 2*(H - z)/(a*(2*z - 1)))/(a*(2*z - 1))\
                                        - 2*((H - z)/(2*z - 1) + 1\
                                        - (H - z)/(a*(2*z - 1)))/(a*(2*z - 1)))/(H - z))
 
def ddAdzz(w,z):
    a = 1
    H = 1
    Kbar = 1
    prefactor = Kbar*2*(1+a)*(1+2*a)/(a**2*H**(1+1/a))
    
    return prefactor*np.where(z < H/2
                              , 4*(H - 2*z)**(1/a)*(-4*z/(H - 2*z) - 3\
                                   + 6*z/(a*(H - 2*z))\
                                   + 3/a - 2*z/(a**2*(H - 2*z)))/(a*(H - 2*z)**2)
                              ,4*(2*z - 1)**(1/a)*(4*(H - z)/(2*z - 1)\
                                  + 3 - 6*(H - z)/(a*(2*z - 1)) - 3/a\
                                  + 2*(H - z)/(a**2*(2*z - 1)))/(a*(2*z - 1)**2))
    
    #return (dAdz(w,z+dz)-dAdz(w,z-dz))/(2*dz)
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
Tmax=1      #Maximum time
dt=1e-3     #Delta time
Ntime=int(Tmax/dt)  #Number of time interval
Np=4000            #Number of particles
Nbins=500           #Number of bin
K=pycnocline        #Define K to pycnocline function
z_e=np.zeros((Ntime,Np))  #initial the history table
z_e[0,:]=np.random.uniform(0.7,0.70001,int(Np))     #Assign initial condition
z_m=z_e.copy()       #copy initialized history table to Milstein scheme
z_v=z_e.copy()       #copy initialized history table to Visser scheme
z_m2=z_e.copy()      #copy initialized history table to Milstein 2nd scheme

start_time = time()

for i in range(Ntime-1):
    z_e[i+1,:]=step_e(z_e[i,:],w,K,dt,Np)
    z_e[i+1,:]=np.where(z_e[i+1,:]>1.0, 1-(z_e[i+1,:]-1),z_e[i+1,:])
    z_e[i+1,:]=np.where(z_e[i+1,:]<0.0, -z_e[i+1,:],z_e[i+1,:])
    
    z_m[i+1,:]=step_m(z_m[i,:],w,K,dt,Np)
    z_m[i+1,:]=np.where(z_m[i+1,:]>1.0, 1-(z_m[i+1,:]-1),z_m[i+1,:])
    z_m[i+1,:]=np.where(z_m[i+1,:]<0.0, -z_m[i+1,:],z_m[i+1,:])
    
    z_v[i+1,:]=step_v(z_v[i,:],w,K,dt,Np)
    z_v[i+1,:]=np.where(z_v[i+1,:]>1.0, 1-(z_v[i+1,:]-1),z_v[i+1,:])
    z_v[i+1,:]=np.where(z_v[i+1,:]<0.0, -z_v[i+1,:],z_v[i+1,:])
    
    z_m2[i+1,:]=step_m2(z_m2[i,:],w,K,dt,Np)
    z_m2[i+1,:]=np.where(z_m2[i+1,:]>1.0, 1-(z_m2[i+1,:]-1),z_m2[i+1,:])
    z_m2[i+1,:]=np.where(z_m2[i+1,:]<0.0, -z_m2[i+1,:],z_m2[i+1,:])
    
    #print(i, end=' ', flush=True)
    
hist1=np.zeros((Nbins-1,Ntime))
hist2=hist1.copy()
hist3=hist1.copy()
hist4=hist1.copy()
bins = np.linspace(0, 1, Nbins)
midpoints = bins[:-1]+(bins[1]-bins[0])/2

times = np.linspace(0, Tmax, Ntime)
#%%
start=5
cmap = plt.get_cmap('jet')
###############
#print Euler  #
###############
for i in range(start,Ntime):
    hist1[:,i-start], bins = np.histogram(z_e[i,:],bins = np.linspace(0, 1, Nbins))
  
fig=plt.figure(1, figsize = (9,5))
#hist1=np.flip(hist1,0)
plt.pcolormesh(times, midpoints, hist1, cmap='jet')
plt.xlabel("Time")
plt.ylabel("Depth")
plt.tight_layout()
plt.savefig("Euler_dt=5e-3.png")
###############
#Print Milstein
###############
for i in range(start,Ntime):
    hist2[:,i-start], bins = np.histogram(z_m[i,:],bins = np.linspace(0, 1, Nbins))

f2=plt.figure(2,figsize = (9,5))
#hist2=np.flip(hist2,0)
plt.pcolormesh(times, midpoints, hist2, cmap='jet')
plt.xlabel("Time")
plt.ylabel("Depth")
plt.tight_layout()
plt.savefig("Mil_1_dt=5e-3.png")

###############
#Print visser
###############
for i in range(start,Ntime):
    hist3[:,i-start], bins = np.histogram(z_v[i,:],bins = np.linspace(0, 1, Nbins))
    
f3=plt.figure(3,figsize = (9,5))
#hist3=np.flip(hist3,0)
plt.pcolormesh(times, midpoints, hist3, cmap='jet')
plt.xlabel("Time")
plt.ylabel("Depth")
plt.tight_layout()
plt.savefig("Visser_2_dt=5e-3.png")

####################
#Print Milstein 2nd#
####################
for i in range(start,Ntime):
    hist4[:,i-start], bins = np.histogram(z_m2[i,:],bins = np.linspace(0, 1, Nbins))
    
f4=plt.figure(4,figsize = (9,5))
#hist4=np.flip(hist4,0)
im=plt.pcolormesh(times, midpoints, hist4, cmap='jet')
plt.xlabel("Time")
plt.ylabel("Depth")
plt.tight_layout()
plt.savefig("Mil_2_dt=5e-3.png")

plt.show()

print("--- %s seconds ---" % (time() - start_time))
#%% The rest is just something to help me check ting correct
#for i in range(Ntime):
#    
#    
#    hist1[:,i], bins = np.histogram(z[:,i], 
#        bins = np.linspace(0, 1, Nbins))
#    
#    hist2[:,i], bins = np.histogram(z_m[:,i], 
#        bins = np.linspace(0, 1, Nbins))
#    
#    midpoints = bins[:-1]+(bins[1]-bins[0])/2
#    
##plt.plot(midpoints,hist,'.')
#f1=plt.figure(1)
#hist1=np.flip(hist1,0)
#plt.imshow(hist1, cmap='jet', interpolation='none')
#
#
#f2=plt.figure(2)
#hist2=np.flip(hist2,0)
#plt.imshow(hist2, cmap='jet', interpolation='none')



#label=np.arange(0,Np)
#plt.plot(label,z,'.')



#z[0]=0.7
#print("Ntime= %d" % Ntime)
#for i in range(Ntime-1):
#    z[i+1]=step(z[i],w,K,dt)
#print(z[-1])
#
#
#times = np.arange(0, Tmax, dt)
#plt.plot(times,z)

#%% Check the diffusitivity profile
#X=np.linspace(0,1,1000)
#Y=np.zeros(1000)
#DK=np.zeros(1000)
#for h in range(X.size):
#    Y[h]=K(X[h])
#    DK[h]=dKdz(X[h])
#f2=plt.figure(2)   
#plt.plot(X,Y)
#f2.show()







































