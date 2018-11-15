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
#import sympy for symbolic derivative
from mpi4py import MPI

comm = MPI.COMM_WORLD
size=comm.Get_size()
rank=comm.Get_rank()
#%%

import sympy
sympy.init_printing()

z = sympy.symbols('z')
K0 = 1e-3# m * * 2 / s
K1 = 6e-3# m / s
Aalpha = 0.5
w=0

sym_Diffu = K0 + K1 * z * sympy.exp(-Aalpha * z)
sym_dKdz = sympy.diff(sym_Diffu, z, 1)
sym_Beta = sympy.sqrt(2 * sym_Diffu)
sym_dBdz = sympy.diff(sym_Beta, z, 1)
sym_ddBdzz = sympy.diff(sym_Beta, z, 2)
sym_Alpha = w + sym_dKdz
sym_dAdz = sympy.diff(sym_Alpha, z, 1)
sym_ddAdzz = sympy.diff(sym_Alpha, z, 2)
sym_dABdz = sympy.diff(sym_Alpha * sym_Beta, z, 1)

Diffu  =  sympy.utilities.lambdify(z,          sym_Diffu,np)
dKdz   =  sympy.utilities.lambdify(z,          sym_dKdz,np)
Beta   =  sympy.utilities.lambdify(z,          sym_Beta,np)
dBdz   =  sympy.utilities.lambdify(z,          sym_dBdz,np)
ddBdzz=  sympy.utilities.lambdify(z,          sym_ddBdzz,np)
Alpha =  sympy.utilities.lambdify(z,      sym_Alpha,np)
dAdz  =  sympy.utilities.lambdify(z,      sym_dAdz,np)
ddAdzz=  sympy.utilities.lambdify(z,      sym_ddAdzz,np)
dABdz =  sympy.utilities.lambdify(z, sym_Alpha*sym_Beta,np)

#%%

########
#Visser#
########
def step_v(z,H,dt,N_sample):
    dW=np.random.uniform(-1,1,N_sample)
    r=1/3
    temp= z + Alpha(z)*dt + np.sqrt(2/r*dt*Diffu(z+dKdz(z)*dt/2))*dW
    temp=np.where(temp<0, -temp ,temp)
    temp=np.where(temp>H, 2*H-temp,temp)
    return temp


def step_v_const(z,H,dt,N_sample):
    K0=3e-3
    dW=np.random.uniform(-1,1,N_sample)
    r=1/3
    temp= z + np.sqrt(2/r*dt*K0)*dW
    temp=np.where(temp<0, -temp ,temp)
    temp=np.where(temp>H, 2*H-temp,temp)
    return temp

##############
#Milstein 1nd#
##############

def step_m(z,H,dt,N_sample):
    dW=np.random.normal(0,np.sqrt(dt),N_sample)
    temp= z + (1/2)*dKdz(z)*(dW*dW+dt) + Beta(z)*dW
    temp=np.where(temp<0, -temp ,temp)
    temp=np.where(temp>H, 2*H-temp,temp)
    return temp


##############
#Milstein 2nd#
##############
def step_m2(z,H,dt,N_sample):
    dW=np.random.normal(0,np.sqrt(dt),N_sample)
    
    temp= z + Alpha(z)*dt+Beta(z)*dW+\
            1/2*Beta(z)*dBdz(z)*(dW*dW-dt)+\
            1/2*(dABdz(z)+1/2*ddBdzz(z)*Beta(z)**2)*dW*dt+\
            1/2*(Alpha(z)*dAdz(z)+1/2*ddAdzz(z)*Beta(z)**2)*dt**2
    
    temp=np.where(temp<0, -temp ,temp)
    temp=np.where(temp>H, 2*H-temp,temp)
    return temp

def step_m2_const(z,H,dt,N_sample):
    K0=3e-3
    dW=np.random.normal(0,np.sqrt(dt),N_sample)
    temp= z +np.sqrt(2*K0)*dW
    temp=np.where(temp<0, -temp ,temp)
    temp=np.where(temp>H, 2*H-temp,temp)
    return temp



#%%
Tmax=12*3600    #Maximum time
dt=1     #Delta time
Ntime=int(Tmax/dt)  #Number of time interval
Np=20000          #Number of particles
Np=int(Np/size)

z_m2=np.random.uniform(0,10,int(Np))
z_m2_Const=z_m2.copy()
z_v=z_m2.copy()   
z_v_const=z_m2.copy()
z_m=z_m2.copy()

H=10
Nregi=10
Gap=int(Ntime/Nregi)

Gap=1           #Set to 1 if you want histogram to recorde all the timestep.
                #Or comment it, then histogram will skip some timestep.
Nbins=1000       #Number of bin

start_time = time()

print("Ntime:  ", Ntime, flush=True)

hist_M2=np.zeros((Nbins-1,),'i')
hist_M2_Const=np.zeros((Nbins-1,),'i')
hist_m=np.zeros((Nbins-1,),'i')
hist_V=np.zeros((Nbins-1,),'i')   
hist_V_Const=np.zeros((Nbins-1,),'i')


Testdepth=1

counter=0
for i in range(Ntime-1):
    z_v=step_v(z_v,H,dt,Np)
    
    z_v_const=step_v_const(z_v_const,H,dt,Np)
    
    z_m=step_m(z_m,H,dt,Np)
    
    z_m2=step_m2(z_m2,H,dt,Np)
    
    z_m2_Const=step_m2_const(z_m2_Const,H,dt,Np)
    
    if (i % int(Ntime/100) ==0 and rank==0):
        print("\r %6.2f"% (i*100/Ntime+1),"%", end="\r",flush=True)

    if ( i % Gap ==0):
        
        temp, _ = np.histogram(z_v, bins = np.linspace(0, Testdepth, Nbins))
        hist_V=hist_V+temp
    
        temp, _ = np.histogram(z_v_const, bins = np.linspace(0, Testdepth, Nbins))
        hist_V_Const=hist_V_Const+temp
        
        temp, _ = np.histogram(z_m, bins = np.linspace(0, Testdepth, Nbins))
        hist_m=hist_m+temp
        
        temp, _ = np.histogram(z_m2, bins = np.linspace(0, Testdepth, Nbins))
        hist_M2=hist_M2+temp
        
        temp, _ = np.histogram(z_m2_Const, bins = np.linspace(0, Testdepth, Nbins))
        hist_M2_Const=hist_M2_Const+temp
        
        counter=counter+1

    
hist_M2=hist_M2/counter

hist_M2_Const=hist_M2_Const/counter
    
hist_V=hist_V/counter
    
hist_V_Const=hist_V_Const/counter

hist_m=hist_m/counter
#%%
comm.barrier()
temp_hist_M2=hist_M2.copy()
temp_hist_M2_Const=hist_M2_Const.copy()
temp_hist_V= hist_V.copy()
temp_hist_V_Const=hist_V_Const.copy()
temp_hist_m=hist_m.copy()

comm.Reduce(temp_hist_V,hist_V,op=MPI.SUM,root=0)
comm.Reduce(temp_hist_V_Const,hist_V_Const,op=MPI.SUM,root=0)
comm.Reduce(temp_hist_M2_Const,hist_M2_Const,op=MPI.SUM,root=0)
comm.Reduce(temp_hist_M2,hist_M2,op=MPI.SUM,root=0)
comm.Reduce(temp_hist_m,hist_m,op=MPI.SUM,root=0)

hist_M2=hist_M2/size

hist_M2_Const=hist_M2_Const/size
    
hist_V=hist_V/size
    
hist_V_Const=hist_V_Const/size

hist_m=hist_m/size
#%%
if (rank==0):
    bins = np.linspace(0, Testdepth, Nbins)
    midpoints = bins[:-1]+(bins[1]-bins[0])/2
    
    
    
    print("\n--- %s seconds ---" % (time() - start_time),end="\n",flush=True)
    
    hist_M2=hist_M2/np.mean(hist_M2)
    hist_M2.dump("M2.dat")
    #np.savetxt('M2.txt', hist_M2, fmt='%.10f')
    
    hist_M2_Const=hist_M2_Const/np.mean(hist_M2_Const)
    hist_M2_Const.dump("M2_const.dat")
    #np.savetxt('M2_const.txt', hist_M2_Const, fmt='%.10f')    
    
    hist_V=hist_V/np.mean(hist_V)
    hist_V.dump("V.dat")
    #np.savetxt('V.txt', hist_V, fmt='%.10f')
    
    hist_V_Const=hist_V_Const/np.mean(hist_V_Const)
    hist_V_Const.dump("V_const.dat")
    #np.savetxt('V_const.txt', hist_V_Const, fmt='%.10f')
    
    hist_m=hist_m/np.mean(hist_m)
    hist_m.dump("M.dat")
    #np.savetxt('M.txt', hist_m, fmt='%.10f')
    
    midpoints.dump("midpoints.dat")
    
    
    fig=plt.figure(1)
    plt.subplot(2,3,1)
    plt.plot(hist_V, midpoints, linewidth=0.5, label= "Visser")
    plt.plot(hist_V_Const, midpoints, linewidth=0.5, label= "Visser_const")
    plt.xlabel('Concetration')
    plt.ylabel('Depth /m')
    plt.title('Visser')
    plt.gca().invert_yaxis()
    
    plt.subplot(2,3,2)
    plt.plot(hist_m, midpoints, linewidth=0.5, label= "Visser")
    plt.plot(hist_M2_Const, midpoints, linewidth=0.5, label= "Visser_const")
    plt.xlabel('Concetration')
    plt.ylabel('Depth /m')
    plt.title('Milstein 1 nd')
    plt.gca().invert_yaxis()
    
    
    plt.subplot(2,3,3)
    plt.plot(hist_M2, midpoints, linewidth=0.5, label= "Milstein_2")
    plt.plot(hist_M2_Const, midpoints, linewidth=0.5, label= "Milstein_2_const")
    plt.xlabel('Concetration')
    plt.ylabel('Depth /m')
    plt.title('Milstein 2 nd')
    plt.gca().invert_yaxis()
    
#    fig=plt.figure(2,figsize=(8,8))
#    x=np.linspace(0,H,100)
#    y=Diffu(x)
#    plt.plot(y,x,label= "Diffusitivity")
#    plt.gca().invert_yaxis()
#    
    plt.show()

