
import numpy as np
import sympy

z, w=sympy.symbols('z w')
H=1
a=1
Kbar = 1
prefactor = Kbar*2*(1+a)*(1+2*a)/(a**2*H**(1+1/a))
First_deriv=    1
Second_deriv=   2

sym_dKdz_down   =  sympy.diff(prefactor*z*(H-2*z)**(1/a),  z, 1)
sym_Beta_down   =  sympy.sqrt(2*prefactor*z*(H-2*z)**(1/a))
sym_dBdz_down   =  sympy.diff(sym_Beta_down,               z, 1)
sym_ddBdzz_down =  sympy.diff(sym_Beta_down,               z, 2)
sym_Alpha_down  =  w + sym_dKdz_down
sym_dAdz_down   =  sympy.diff(sym_Alpha_down,              z, 1)
sym_ddAdzz_down =  sympy.diff(sym_Alpha_down,              z, 2)
sym_dABdz_down  =  sympy.diff(sym_Alpha_down*sym_Beta_down,z, 1)

sym_dKdz_up     =  sympy.diff(prefactor* (H-z)*(2*z-1)**(1/a), z, 1)
sym_Beta_up     =  sympy.sqrt(2*prefactor* (H-z)*(2*z-1)**(1/a))
sym_dBdz_up     =  sympy.diff(sym_Beta_up,                     z, 1)
sym_ddBdzz_up   =  sympy.diff(sym_Beta_up,                     z, 2)
sym_Alpha_up    =  w + sym_dKdz_up
sym_dAdz_up     =  sympy.diff(sym_Alpha_up,                    z, 1)
sym_ddAdzz_up   =  sympy.diff(sym_Alpha_up,                    z, 2)
sym_dABdz_up    =  sympy.diff(sym_Alpha_up*sym_Beta_up,        z, 1)

dKdz_down   =  sympy.utilities.lambdify(z,          sym_dKdz_down,np)
Beta_down   =  sympy.utilities.lambdify(z,          sym_Beta_down,np)
dBdz_down   =  sympy.utilities.lambdify(z,          sym_dBdz_down,np)
ddBdzz_down =  sympy.utilities.lambdify(z,          sym_ddBdzz_down,np)
Alpha_down  =  sympy.utilities.lambdify([w,z],      sym_Alpha_down,np)
dAdz_down   =  sympy.utilities.lambdify([w,z],      sym_dAdz_down ,np)
ddAdzz_down =  sympy.utilities.lambdify([w,z],      sym_ddAdzz_down,np)
dABdz_down  =  sympy.utilities.lambdify([w,z], sym_Alpha_down*sym_Beta_down,np)

dKdz_up   =  sympy.utilities.lambdify( z,     sym_dKdz_up,                np)
Beta_up   =  sympy.utilities.lambdify( z,     sym_Beta_up,                np)
dBdz_up   =  sympy.utilities.lambdify( z,     sym_dBdz_up,                np)
ddBdzz_up =  sympy.utilities.lambdify( z,     sym_ddBdzz_up,              np)
Alpha_up  =  sympy.utilities.lambdify( [w,z], sym_Alpha_up,               np)
dAdz_up   =  sympy.utilities.lambdify( [w,z], sym_dAdz_up,                np)
ddAdzz_up =  sympy.utilities.lambdify( [w,z], sym_ddAdzz_up,              np)
dABdz_up  =  sympy.utilities.lambdify( [w,z], sym_Alpha_up*sym_Beta_up,   np)

#%%
#Some ultility function


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
K=pycnocline

def dKdz(z):
    return np.where(z < H/2,  dKdz_down(z), dKdz_up(z))

def alpha(w,z):
    return np.where(z < H/2,  w+dKdz_down(z),w+dKdz_up(z))

def dAdz(w,z):
    return np.where(z < H/2, dAdz_down(w,z),dAdz_up(w,z))

def ddAdzz(w,z):
    return np.where(z < H/2, ddAdzz_down(w,z),ddAdzz_up(w,z))

def beta(z):
    mask=(z<H/2)
    temp=np.empty_like(z);
    temp[mask]=Beta_down(z[mask])
    temp[~mask]=Beta_up(z[~mask])
    return temp
    #return np.where(z < H/2, Beta_down(z),Beta_up(z))

def dBdz(z):
    mask=(z<H/2)
    temp=np.empty_like(z);
    temp[mask]=dBdz_down(z[mask])
    temp[~mask]=dBdz_up(z[~mask])
    return temp


def ddBdzz(z):
    mask=(z<H/2)
    temp=np.empty_like(z);
    temp[mask]=ddBdzz_down(z[mask])
    temp[~mask]=ddBdzz_up(z[~mask])
    return temp
    #return np.where(z < H/2, ddBdzz_down(z),ddBdzz_up(z))

def dABdz(w,z):
    mask=(z<H/2)
    temp=np.empty_like(z);
    temp[mask]=dABdz_down(w,z[mask])
    temp[~mask]=dABdz_up(w,z[~mask])
    return temp
    #return np.where(z < H/2, dABdz_down(w,z),dABdz_up(w,z))

#%%
#Calculate nest step

#######
#Euler#
#######
def step_e(z,w,dt,N_sample):

    dW=np.random.normal(0,np.sqrt(dt),N_sample)
    temp=z+alpha(w,z)*dt+beta(z)*dW
    temp=np.where(temp>1.0, 1-(temp-1),temp)
    temp=np.where(temp<0.0, -temp,temp)
    return temp

##########
#Milstein#
##########
def step_m(z,w,dt,N_sample):
    dW=np.random.normal(0,np.sqrt(dt),N_sample)
    temp= z + w*dt + (1/2)*dKdz(z)*(dW*dW+dt) + beta(z)*dW
    temp=np.where(temp>1.0, 1-(temp-1),temp)
    temp=np.where(temp<0.0, -temp,temp)
    return temp


########
#Visser#
########
def step_m2(z,w,dt,N_sample):
    dW=np.random.normal(0,np.sqrt(dt),N_sample)
    
    temp= z + alpha(w,z)*dt+beta(z)*dW+\
            1/2*beta(z)*dBdz(z)*(dW*dW-dt)+\
            1/2*(dABdz(w,z)+1/2*ddBdzz(z)*beta(z)**2)*dW*dt+\
            1/2*(alpha(w,z)*dAdz(w,z)+1/2*ddAdzz(w,z)*beta(z)**2)*dt**2
    
    temp=np.where(temp>1.0, 1-(temp-1),temp)
    temp=np.where(temp<0.0, -temp,temp)
    return temp

##############
#Milstein 2nd#
##############
def step_v(z,w,dt,N_sample):
    # dW=np.random.normal(0,np.sqrt(dt),N_sample)
    # temp= z + alpha(w,z)*dt + beta(z + 1/2*alpha(w,z)*dt)*dW
    dW=np.random.uniform(-1,1,N_sample)
    r=1/3
    temp=z + alpha(w,z)*dt + dW*np.sqrt(2/r*dt*K(z+dKdz(z)*dt/2) )
    
    temp=np.where(temp>1.0, 1-(temp-1),temp)
    temp=np.where(temp<0.0, -temp,temp)
    return temp
