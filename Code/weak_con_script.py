# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 19:50:38 2018

@author: mauhi
"""
#%%
from matplotlib import pyplot as plt
plt.style.use('bmh')

import numpy as np

import sys

#from time import time
#%%
import sympy
sympy.init_printing()

z = sympy.symbols('z')
H=25.0
K0=5e-3
K1=1e-3
turncation=H/2
scale=1

sym_Diffu = K0+(K1-K0)*(1-1/(1+sympy.exp(-(turncation-z)/scale)))
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

def Gaussian(z, mu, sigma):
    return 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-((z-mu)/sigma)**2/2)
    
def OneTimeStep(C, K, dKdz, dt, dz, w):
    dC=np.empty(C.shape[0]+2, dtype=float)
     #We need two extra ghost points at the start and the end.   
    NyC=np.empty(C.shape[0]+2, dtype=float)
    
    NyC[1:-1]=C.copy()
    NyC[0]=C[1]
    NyC[-1]=w*C[-1]/K[-1]*dz+C[-2]
    
    
    dC=dKdz*(NyC[2:]-NyC[0:-2])/(2*dz)\
            +K*(NyC[2:]-2*NyC[1:-1]+NyC[0:-2])/(dz**2)+\
            -w*(NyC[2:]-NyC[0:-2])/(2*dz)

    dC=dt*dC;
    C=C+dC
    return C

def step_m2(z,H,dt,N_sample):
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


#%%
#input parameters.
dz=0.04
H=25.0
dtEu=0.1
dtLa=1200  #dt for particle scheme
T=12*3600 #one hours
w=0
Np=2000000             #Number of particles

#Fixed parameters in the problem
mu=10.0
sigma=1.0

#Dependent parameter.
Nt=int(T/dtEu) 
Ntp=int(T/dtLa)
Nz=int(H/dz)
if (T%dtLa) != 0:
    print("Tmax is not dividable to dtLa")
    sys.exit(0)

zEu=np.linspace(0,H,Nz)    
Conc=Gaussian(zEu,mu,sigma) 
ConcInit=Conc.copy()

K_array=Diffu(zEu)
dKdz_arry=dKdz(zEu)

plt.plot(zEu,K_array)
plt.show()
#%%
particles=np.random.normal(mu, sigma, Np)
Num_O_Bins_Poi=50
bins=np.linspace(0,25,Num_O_Bins_Poi)

hist_start,_=np.histogram(particles, bins=bins)

#for i in range(Nt):
#    Conc=OneTimeStep(Conc, K_array, dKdz_arry, dtEu, dz, w)
#    
#    if (i % int(Nt/100) ==0): #Just for printing
#        print("\r", int(i*100/Nt+1),"%", end="\r",flush=True)
#print("\n")
#
#np.save("Concentration", Conc)

Conc=np.load("Concentration.npy")
print("Number of time step: ", Ntp)
print("Number of particles: ", Np)
for i in range(Ntp):
    particles=step_m2(particles,H,dtLa,Np)    
    print("\r",i," of ",Ntp, end="\r", flush=True)


hist,_=np.histogram(particles, bins=bins)
midpoints=bins[0:-1]+(bins[1]-bins[0])/2
#Normalize hist
hist_start=hist_start/(np.sum(hist_start)*(bins[1]-bins[0]))
hist=hist/(np.sum(hist)*(bins[1]-bins[0]))
###


fig=plt.figure(1)
plt.plot(zEu,Conc)   
plt.plot(midpoints,hist)
plt.show()

#%%
"""
1. Calculate the "$order" momentum of Eulerian.
2. Calculate the "$order" momentum of Lagrangian.
"""
print("---Second part---")
print("dt of Lagrangian: ", dtLa)
order=1
momEu=np.sum(Conc*zEu**order)*dz
momLagM2=np.sum(particles**order)/particles.size;
weak_con=np.abs(momLagM2-momEu)

print("Order: ", order)
print("Weak convergence difference: ", weak_con)
print("Done")

#%%
"""
Junk code


#Calcutate average concentration in bins
avgc=np.zeros(0)
avgC=np.zeros(0)
####
for i in range (len(midpoints)):
    
    mask_min= (z>=bins[i])
    mask_max= (z<=bins[i+1])   #Here because the equal-like.
    mask= (mask_min==mask_max)
    
    local_C= C[mask]
    avgC= np.append(avgC,np.sum(local_C)/len(local_C))
###
    
#Calculate RMSE
RMSE= np.sqrt( np.sum(np.power((hist-avgC),2))  /(len(midpoints)))

fig=plt.figure(2)
plt.plot(midpoints,avgC)
plt.plot(midpoints,hist)
plt.show() 

print("RMSE: ", RMSE)

#fig=plt.figure(1)
#plt.plot(z,c)   
#plt.plot(z,C)
#plt.plot(midpoints,hist_start)
#plt.plot(midpoints,hist)
#plt.show()

#sym_Beta = sympy.sqrt(2 * sym_Diffu)
#sym_dBdz = sympy.diff(sym_Beta, z, 1)
#sym_ddBdzz = sympy.diff(sym_Beta, z, 2)
#sym_Alpha = w + sym_dKdz
#sym_dAdz = sympy.diff(sym_Alpha, z, 1)
#sym_ddAdzz = sympy.diff(sym_Alpha, z, 2)
#sym_dABdz = sympy.diff(sym_Alpha * sym_Beta, z, 1)


#Beta   =  sympy.utilities.lambdify(z,          sym_Beta,np)
#dBdz   =  sympy.utilities.lambdify(z,          sym_dBdz,np)
#ddBdzz=  sympy.utilities.lambdify(z,          sym_ddBdzz,np)
#Alpha =  sympy.utilities.lambdify(z,      sym_Alpha,np)
#dAdz  =  sympy.utilities.lambdify(z,      sym_dAdz,np)
#ddAdzz=  sympy.utilities.lambdify(z,      sym_ddAdzz,np)
#dABdz =  sympy.utilities.lambdify(z, sym_Alpha*sym_Beta,np)


#z = sympy.symbols('z')
#K0 = 1e-3# m * * 2 / s
#K1 = 6e-3# m / s
#Aalpha = 0.5
#w=0
#
#sym_Diffu = K0 + K1 * z * sympy.exp(-Aalpha * z)


"""


























#plt.plot(z,K)
#plt.plot(z,dKdz)
