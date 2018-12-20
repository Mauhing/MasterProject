#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 21:05:23 2018

@author: mhyip
"""

from matplotlib import pyplot as plt
plt.style.use('bmh')
import numpy as np
import numpy.matlib
import numpy.linalg
import sympy
import sys  #Use this to abort if the condition is not right.
from time import time
import scipy
from scipy import stats
from scipy import sparse
from scipy.sparse import linalg
from scipy.sparse.linalg import spsolve
from scipy.stats import norm
#from numba import jit, njit, prange
#import numba
#%%
sympy.init_printing()

z = sympy.symbols('z')
H=25.0
K0=1e-6
K1=1e-4
#K0=5e-3
#K1=1e-3S

turncation=H/2
scale=1.0

#sym_Diffu = K0+(K1-K0)*(1-1/(1+sympy.exp(-(turncation-z)/scale)))
sym_Diffu = K0+(K1-K0)/(sympy.exp(scale*(z-turncation))+1)
sym_dKdz = sympy.diff(sym_Diffu, z, 1)
sym_d2Kdz2 =  sympy.diff(sym_Diffu, z, 2)
sym_d3Kdz3   =  sympy.diff(sym_Diffu, z, 3)

Diffu  =  sympy.utilities.lambdify(z,sym_Diffu,np)
dKdz   =  sympy.utilities.lambdify(z,sym_dKdz,np)
d2Kdz =  sympy.utilities.lambdify(z,sym_d2Kdz2,np)
d3Kdz = sympy.utilities.lambdify(z,sym_d3Kdz3,np)

del z #delete the symbol

print("Sympy finished")


#%% Functions

def Gaussian(z, mu, sigma):
    return 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-((z-mu)/sigma)**2/2)


def OneTimeStep(C, K, dKdz, dt, dz):
    dC=np.empty(C.shape[0]+2, dtype=float)
     #We need two extra ghost points at the start and the end.   
    NyC=np.empty(C.shape[0]+2, dtype=float)
    NyC[1:-1]=C.copy()
    NyC[0]=C[1]
    NyC[-1]=C[-2]
    
    dC=dKdz*(NyC[2:]-NyC[0:-2])/(2*dz)+K*(NyC[2:]-2*NyC[1:-1]+NyC[0:-2])/(dz**2)
            
    dC=dt*dC;
    C=C+dC
    return C


def step_m2(z,H,dt,N_sample,w=0):
    dW=np.random.normal(0,np.sqrt(dt),N_sample)
    
    k=Diffu(z)
    dkdz=dKdz(z)
    ddkdz=d2Kdz(z)
    dddkdz=d3Kdz(z)
    sqrt2k=np.sqrt(2*k)
    
    a= w + dkdz
    da=ddkdz
    dda=dddkdz
    b= sqrt2k 
    db=dkdz/b
    ddb=ddkdz/b - ((dkdz)**2)/b**3
    ab=da*b+a*db
    
    temp= z + a*dt+b*dW+1/2*b*db*(dW*dW-dt)+1/2*(ab+1/2*ddb*b**2)*dW*dt+\
            1/2*(a*da+1/2*dda*b**2)*dt**2
            
    temp=np.where(temp<0, -temp ,temp)
    temp=np.where(temp>H, 2*H-temp,temp)
    return temp

def step_e(z,H,dt,N_sample,w=0):
    dW=np.random.normal(0,np.sqrt(dt),N_sample) 
    
    a=w+dKdz(z)
    b=np.sqrt(2*Diffu(z))
    temp=z+a*dt+b*dW
    temp=np.where(temp<0, -temp ,temp)
    temp=np.where(temp>H, 2*H-temp,temp)
    return temp

def step_m(z,H,dt,N_sample,dW=None,w=0):
    if dW is None:
        dW=np.random.normal(0,np.sqrt(dt),N_sample)
    
    k=Diffu(z)
    dkdz=dKdz(z)
    sqrt2k=np.sqrt(2*k)
    
    a= w + dkdz
    b= sqrt2k 
    db=dkdz/b
    #temp= z + w*dt + (1/2)*dkdz*(dW*dW+dt) + b*dW
    temp= z+ a*dt+1/2*(b*db)*(dW*dW-dt)+b*dW
    
    temp=np.where(temp<0, -temp ,temp)
    temp=np.where(temp>H, 2*H-temp,temp)
    return temp

def step_v(z,H,dt,N_sample,dW=None,w=0):
    if dW is None:
        dW=np.random.normal(0,np.sqrt(dt),N_sample)
    a=w+dKdz(z)
    b=np.sqrt(2*Diffu(z + 1/2*a*dt))
    temp= z + a*dt + b*dW
    temp=np.where(temp<0, -temp ,temp)
    temp=np.where(temp>H, 2*H-temp,temp)
    return temp
#%% Functions

def Lagrangian(H=None, dtLa=None, T=None, Np=None, mu=None, sigma=None):
    if (H or dtLa or T or Np or mu or sigma) == None:
        print("Using default values")
        H=25.0
        dtLa=1200  
        T=0.5*3600 
        Np=2000000             

    Nt=int(T/dtLa)
    
    if (T%dtLa) != 0:
        print("Tmax is not dividable to dtLa")
        sys.exit(0)

    LaM2=np.random.normal(mu, sigma, Np)
    LaM1=LaM2.copy()
    LaVi=LaM2.copy()
    LaEM=LaM2.copy()
    
    start=time()
    print("---Lagrangian part---")
    print("Number of time step: ", Nt)
    print("Number of particles: ", Np)
    print("dt of Lagrangian: ", dtLa)
    for i in range(Nt):
        print("\r","Working: ", i, " of ", Nt, end="\r", flush=True)       
        LaM2=step_m2(LaM2,H,dtLa,Np)
        LaEM=step_e(LaEM,H,dtLa,Np)
        LaVi=step_v(LaVi,H,dtLa,Np)
        LaM1=step_m(LaM1,H,dtLa,Np)

    """
    1. Calculate the "$order" momentum of Eulerian.
    2. Calculate the "$order" momentum of Lagrangian.
    """
    order=1
    momLagM2=np.sum(LaM2**order)/LaM2.size
    momLagEM=np.sum(LaEM**order)/LaEM.size
    momLagVi=np.sum(LaVi**order)/LaVi.size
    momLagM1=np.sum(LaM1**order)/LaM1.size
    
    print("Order: ", order)
    print("Done! Time used: ", time()-start)
    return momLagM2,momLagEM,momLagVi,momLagM1

def Lagrangian_test(H=None, dtLa=None, T=None, Np=None, mu=None, sigma=None):
    if (H or dtLa or T or Np or mu or sigma) == None:
        print("Using default values")
        H=25.0
        dtLa=1200  
        T=0.5*3600 
        Np=2000000             

    Nt=int(T/dtLa)
    
    if (T%dtLa) != 0:
        print("Tmax is not dividable to dtLa")
        sys.exit(0)

    LaM2=np.random.normal(mu, sigma, Np)
    LaM1=LaM2.copy()
    LaVi=LaM2.copy()
    LaEM=LaM2.copy()
    

    print("---Lagrangian part---")
    print("Number of time step: ", Nt)
    print("Number of particles: ", Np)
    print("dt of Lagrangian: ", dtLa)
    for i in range(Nt):
        print("\r","Working: ", i, " of ", Nt, end="\r", flush=True)       
        LaM2=step_m2(LaM2,H,dtLa,Np)
        LaEM=step_e(LaEM,H,dtLa,Np)
        LaVi=step_v(LaVi,H,dtLa,Np)
        LaM1=step_m(LaM1,H,dtLa,Np)

    return LaEM, LaVi, LaM1, LaM2

def Eulerian(dz, H, Concentration):
    Nz=int(H/dz)
    zEu=np.linspace(0,H,Nz)
    order=1
    momConc=np.sum(Concentration*zEu**order)*dz
    
    return momConc

def Eulerian_Concentration(dz=None, H=None, dtEu=None, T=None, mu=None, sigma=None):
    if (dz or H or dtEu or T) is None:
        print ("Using default value")
        dz=0.04
        H=25.0
        dtEu=0.1        
        T=12*3600 
        
    Nt=int(T/dtEu) 
    Nz=int(H/dz)
    zEu=np.linspace(0,H,Nz)    
    Conc=Gaussian(zEu,mu,sigma) 
    K_array=Diffu(zEu)
    dKdz_arry=dKdz(zEu)
        
    for i in range(Nt):
        Conc=OneTimeStep(Conc, K_array, dKdz_arry, dtEu, dz)
        if (i % int(Nt/100) ==0): #Just for printing
            print("\r", float(i*100/Nt+1),"%", end="\r",flush=True)
    print("\n")
    
    return Conc


def crank_nicolson(dz, H, dt, T, mu, sigma):
    
    Nt=int(T/dt) 
    Nz=int(H/dz)
    zEu=np.linspace(0,H,Nz)    
    Concentration=Gaussian(zEu,mu,sigma) 
    K_array=Diffu(zEu)
    dKdz_array=dKdz(zEu)
    
    #A*C_{n+1}=B*C_{n} where A and B is matrix
    #A is equal I + Beta + Alpha
    #B is equal I - Beta - Alpha
    
    alpha_upper=K_array[1:]*dt/(dz**2)  #upper diagonal array
    alpha_mid=K_array*dt/(dz**2)        #main diagonal array
    alpha_lower=K_array[:-1]*dt/(dz**2) #lower diagonal array
    
    Alpha=numpy.matlib.zeros((Nz,Nz))
    fill=np.arange(Nz)
    Alpha[fill,fill]=alpha_mid          #assign diagonal array
    fill=np.arange(Nz-1)
    Alpha[fill,fill+1]=-alpha_upper/2   #assign diagonal array
    Alpha[fill+1,fill]=-alpha_lower/2   #assign diagonal array
    
    beta_upper=dKdz_array[1:]*dt/(2*dz) #upper diagonal array
    beta_lower=dKdz_array[:-1]*dt/(2*dz)#lower diagonal array
    
    Beta=numpy.matlib.zeros((Nz,Nz))
    fill=np.arange(Nz-1)
    Beta[fill,fill+1]=-beta_upper/2 #assign diagonal array
    Beta[fill+1,fill]=beta_lower/2  #assign diagonal array
    
    I=numpy.matlib.zeros((Nz,Nz)) #Identity matrix
    np.fill_diagonal(I,1.0)
    
    
    #A*C_{n+1}=B*C_{n}
    A=I+Beta+Alpha
    B=I-Beta-Alpha
    
    A[0,1]=-alpha_upper[0]      #assign Nuemann bounday condition
    A[-1,-2]=-alpha_lower[-1]   #assign Nuemann bounday condition
    
    B[0,1]=alpha_upper[1]       #assign Nuemann bounday condition
    B[-1,-2]=alpha_lower[-2]    #assign Nuemann bounday condition
    
    
    #C_{n+1}=A^{-1} * B*C_{n} and let C=A^{-1} * B
    InvA=numpy.linalg.inv(A)
    C=InvA*B
    
    del A #Big matrix, free memory
    del B #Big matrix, free memory
    

    #Convert the array to column vector.
    Concentration=(np.matrix(Concentration)).T

#   Two ways to do this:
#    1) 
#    first way, very fast way
#    Matrix-multiplication is assosiative 
    D=numpy.linalg.matrix_power(C,Nt)
    Concentration=D*Concentration
    
#    seconde way, very slow way  
#    2)
#    print("Nt is: ", Nt)
#    for i in range(Nt):
#        Concentration=C*Concentration
#        if (i % int(Nt/100) ==0): #Just for printing
#            print("\r", float(i*100/Nt+1),"%", end="\r",flush=True)
#    print("\n")
    
    Concentration=Concentration.T
    
    return Concentration.A1


#%%

#input parameters.
dz=0.01
dtEu=0.01
T=12*3600               #12 hours
Np=2000000              #Number of particles
mu=12.5
sigma=1.0

concentration=crank_nicolson(dz, H, dtEu, T, mu, sigma)
#%%
dz=0.01
dtEu=0.01
T=12*3600               #12 hours
Np=2000000              #Number of particles
mu=12.5
sigma=1.0

Nt=int(T/dtEu) 
Nz=int(H/dz)
zEu=np.linspace(0,H,Nz)    
Conc=Gaussian(zEu,mu,sigma) 
K_array=Diffu(zEu)
dKdz_arry=dKdz(zEu)
    
for i in range(Nt):
    Conc=OneTimeStep(Conc, K_array, dKdz_arry, dtEu, dz)
    if (i % int(Nt/100) ==0): #Just for printing
        print("\r", float(i*100/Nt+1),"%", end="\r",flush=True)
print("\n")
    
#concentrationEu=Eulerian_test(dz, H, 0.001, T, mu, sigma)
#%%
z=np.linspace(0,25, concentration.size)
plt.plot(z,concentration)
plt.plot(z,Conc)

#%%
#Part 2 Calculate weak convergence

z=np.linspace(0,25, 500)
D=Diffu(z)
plt.plot(z,D,"-")
plt.show()
z=np.linspace(0,25, 2000)
D=np.amin(np.abs(1/d2Kdz(z)))
print(D, "\n")
#%%
#input parameters.
dz=0.005
dtEu=0.001
T=12*3600               #12 hours
Np=20000000              #Number of particles
mu=12.5
sigma=1.0

Nt=int(T/dtEu) 
Nz=int(H/dz)

#first calculate eulerian part
Concentration=Eulerian_Concentration(dz, H, dtEu, T, mu, sigma)
#np.save("Tor_Concentration", Concentration)
#%%
Concentration=np.load("Tor_Concentration.npy")
momConc=Eulerian(dz, H, Concentration)
#%%
Concentration=np.load("Tor_Concentration.npy")
#dtLa_array=np.array([100, 200, 400, 600, 800, 1200, 1600, 1800, 2400, 2700])
dtLa_array=np.array([ 800, 1200, 2400, 3600, 2*3600, 3*3600, 4*3600, 6*3600])
#dtLa_array=np.array([600,900,1200, 1800, 2700, 3600, 5400, 7200, 10800, 14400, 21600])

weak_conM2=np.zeros(dtLa_array.size)
weak_conEM=np.zeros(dtLa_array.size)
weak_conVi=np.zeros(dtLa_array.size)
weak_conM1=np.zeros(dtLa_array.size)

momLagM2=np.zeros(dtLa_array.size)
momLagEM=np.zeros(dtLa_array.size)
momLagM1=np.zeros(dtLa_array.size)
momLagVi=np.zeros(dtLa_array.size)


for i in range(dtLa_array.size):
    momLagM2[i],momLagEM[i],momLagVi[i],momLagM1[i]=Lagrangian(H, dtLa_array[i], T, Np, mu, sigma)
    weak_conM2[i]=np.abs(momLagM2[i]-momConc)
    weak_conEM[i]=np.abs(momLagEM[i]-momConc)
    weak_conVi[i]=np.abs(momLagVi[i]-momConc)
    weak_conM1[i]=np.abs(momLagM1[i]-momConc)
    print("Tor_weak_conM2", weak_conM2)
    print("Tor_weak_conEM", weak_conEM)
    print("Tor_weak_conM1", weak_conM1)
    print("Tor_weak_conVi", weak_conVi)
    print("")
    
np.save("Tor_momLagEM",momLagEM)
np.save("Tor_momLagVi",momLagVi)
np.save("Tor_momLagM1",momLagM1)
np.save("Tor_momLagM2",momLagM2)
    
np.save("Tor_Weak_EM",weak_conEM)
np.save("Tor_Weak_M2",weak_conM2)
np.save("Tor_Weak_Vi",weak_conVi)
np.save("Tor_Weak_M1",weak_conM1)
np.save("Tor_dtArray",dtLa_array)


#import winsound
#duration = 1000  # millisecond
#freq = 600  # Hz
#winsound.Beep(freq, duration)
#%%

#%%
dz=0.005
dtEu=0.001
T=12*3600               #12 hours
Np=2000000              #Number of particles
mu=12.5
sigma=1.0

EM, Vi, M1, M2=Lagrangian_test(H, 300, T, Np, mu, sigma)
BigEM, BigVi, BigM1, BigM2=Lagrangian_test(H, 2700, T, Np, mu, sigma)
#%%
momEM=np.sum(EM)/EM.size
momVi=np.sum(Vi)/Vi.size
momM1=np.sum(M1)/M1.size
momM2=np.sum(M2)/M2.size

momBigEM=np.sum(BigEM)/BigEM.size
momBigVi=np.sum(BigVi)/BigVi.size
momBigM1=np.sum(BigM1)/BigM1.size
momBigM2=np.sum(BigM2)/BigM2.size
#%%
z=np.linspace(0,25, Concentration.size)
D=np.amin(np.abs(1/d2Kdz(z)))
print(D, "\n")

fig=plt.figure(1, figsize=(8,5))
fig, ax = plt.subplots(1)
z=np.linspace(0, 25, Concentration.size)
plt.plot(z,Concentration)
n, bins, patches=ax.hist(EM, 1000, density=1)

fig=plt.figure(2, figsize=(8,5))
fig, ax = plt.subplots(1)
z=np.linspace(0, 25, Concentration.size)
plt.plot(z,Concentration)
n, bins, patches=ax.hist(Vi, 1000, density=1)

fig=plt.figure(3, figsize=(8,5))
fig, ax = plt.subplots(1)
z=np.linspace(0, 25, Concentration.size)
plt.plot(z,Concentration)
n, bins, patches=ax.hist(M1, 1000, density=1)

fig=plt.figure(4, figsize=(8,5))
fig, ax = plt.subplots(1)
z=np.linspace(0, 25, Concentration.size)
plt.plot(z,Concentration)
n, bins, patches=ax.hist(M2, 1000, density=1)
