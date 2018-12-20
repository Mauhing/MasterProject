#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 15:43:46 2018

@author: mhyip
"""

from matplotlib import pyplot as plt
plt.style.use('bmh')
import numpy as np
import numpy.matlib
import sympy
import sys  #Use this to abort if the condition is not right.
from time import time
import scipy

from scipy import stats
from scipy import sparse
from scipy.sparse import linalg
from scipy.sparse.linalg import spsolve
from scipy.stats import norm
#%%

sympy.init_printing()

z = sympy.symbols('z')
H=25.0
K0=1e-4
K1=1e-6
turncation=H/2
scale=1.0
sym_Diffu = K0+(K1-K0)*(1-1/(1+sympy.exp(-(turncation-z)/scale)))

#K0=1e-6
#K1=1e-4
#sym_Diffu = K0+(K1-K0)/(sympy.exp(scale*(z-turncation))+1)


sym_dKdz = sympy.diff(sym_Diffu, z, 1)
sym_d2Kdz2 =  sympy.diff(sym_Diffu, z, 2)
sym_d3Kdz3   =  sympy.diff(sym_Diffu, z, 3)

Diffu  =  sympy.utilities.lambdify(z,sym_Diffu,np)
dKdz   =  sympy.utilities.lambdify(z,sym_dKdz,np)
d2Kdz =  sympy.utilities.lambdify(z,sym_d2Kdz2,np)
d3Kdz = sympy.utilities.lambdify(z,sym_d3Kdz3,np)

del z #delete the symbol

print("Sympy finished")
#%%
#z=np.linspace(0, H, 2500)
#plt.plot(z, Diffu(z))

#%% Functions

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


def crankNicolson(C0, D0, X, dt, Tmax):
    # Numerical parameters
    dx = X[1] - X[0]
    Nx = X.size
    Nt = int(Tmax / dt)
    a  = dt/(8*dx**2)
    b  = dt/(4*dx)
    
    # Problem arrays
    # Pad with halo cells and 
    # copy elements onto boundary
    C      = np.zeros((Nt+1, Nx))
    C[0,:] = C0
    # Diffusivity (here constant in time)
    D      = np.zeros(Nx)
    D[:]   = D0
    # Create system matrices
    # Equation is L C_i = R C_{i+1}
    # Left hand side
    L = np.zeros((Nx, Nx))
    # Fill non-boundary points first
    L[1:-1,1:-1] += np.diag(1 - 8*a*D[1:-1], 0)
    L[:-1,:-1] += np.diag(-a*(D[2:]-D[:-2]) + 4*a*D[ 1:-1], -1)
    L[1:,1:] += np.diag( a*(D[2:]-D[:-2]) + 4*a*D[1:-1], +1)
    # No-diffusive-flux BC at top
    L[ 0, 0] = 1 - 8*a*D[0]
    L[ 0, 1] = 8*a*D[0]
    # No-diffusive-flux  BC at bottom
    L[-1,-1] = 1 - 8*a*D[-1]
    L[-1,-2] = 8*a*D[-1]
    # Convert to sparse matrix
    L  = scipy.sparse.csr_matrix(L)
    # Right hand side
    R = np.zeros((Nx, Nx))
    # Fill non-boundary points first
    R[1:-1,1:-1] += np.diag(1 + 8*a*D[1:-1], 0)
    R[:-1,:-1] += np.diag( a*(D[2:]-D[:-2]) - 4*a*D[ 1:-1], -1)
    R[1:,1:] += np.diag(-a*(D[2:]-D[:-2]) - 4*a*D[1:-1], +1)
    # No-diffusive-flux  BC at top
    R[ 0, 0] = 1 + 8*a*D[0]
    R[ 0, 1] = -8*a*D[0]
    # No-diffusive-flux  BC at bottom
    R[-1,-1] = 1 + 8*a*D[-1]
    R[-1,-2] = -8*a*D[-1]
    # Convert to sparse matrix
    R  = scipy.sparse.csr_matrix(R)
    C=C[0,:]
    for t in range(Nt):
        x = L.dot(C)
        C = spsolve(R, x)
    return C


#%%
H=25
dt2=10
dt1=5
T=12*3600
mu=H/2
sigma=1
Np=20000000
Nz=2500

zArray,dz=np.linspace(0,H,Nz,retstep=True)
D0=Diffu(zArray)
Conc=norm(loc=mu, scale=sigma).pdf(zArray)

timestep=np.array([10, 5, 2])
momentum_crank=np.zeros(timestep.size)
for i in range(timestep.size):
    concentration=crankNicolson(Conc, D0, zArray, timestep[i], T)
    momentum_crank[i]=np.sum(concentration*zArray)*dz
    print("With dt:=",timestep[i], ", the momentum is:",momentum_crank[i])
momConc=momentum_crank[-1]

#%%
z=np.linspace(0,25, 500)
D=Diffu(z)
#plt.plot(z,D,"-")
plt.show()
z=np.linspace(0,25, 2000)
D=np.amin(np.abs(1/d2Kdz(z)))
print(D, "\n")


#dtLa_array=np.array([ 400, 600, 800, 1200, 1600, 1800, 2400, 2700])
#dtLa_array=np.array([ 800, 1200, 2400, 3600, 2*3600, 3*3600, 4*3600, 6*3600])
#dtLa_array=np.array([600,900,1200, 1800, 2700, 3600, 5400, 7200, 10800, 14400, 21600])
dtLa_array=np.array([ 3600, 4320, 4800,5400,2*3600, 8640, 10800, 14400, 21600])
#dtLa_array=np.array([100, 200, 400, 600, 800, 1200, 1600, 1800, 2400, 2700])

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
    print("weak_conM2", weak_conM2)
    print("weak_conEM", weak_conEM)
    print("weak_conM1", weak_conM1)
    print("weak_conVi", weak_conVi)
    print("")
    
np.save("momLagEM",momLagEM)
np.save("momLagVi",momLagVi)
np.save("momLagM1",momLagM1)
np.save("momLagM2",momLagM2)
    
np.save("Weak_EM",weak_conEM)
np.save("Weak_M2",weak_conM2)
np.save("Weak_Vi",weak_conVi)
np.save("Weak_M1",weak_conM1)
np.save("dtArray",dtLa_array)


#import winsound
#duration = 1000  # millisecond
#freq = 600  # Hz
#winsound.Beep(freq, duration)
