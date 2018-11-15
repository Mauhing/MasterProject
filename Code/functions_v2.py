
import numpy as np
import sympy

z, w=sympy.symbols('z w')
H=1
a=1
Kbar = 1
prefactor = Kbar*2*(1+a)*(1+2*a)/(a**2*H**(1+1/a))

sym_dKdz_down   =  sympy.diff(prefactor*z*(H-2*z)**(1/a),  z, 1)
sym_d2Kdz2_down   =  sympy.diff(prefactor*z*(H-2*z)**(1/a),  z, 2)
sym_d3Kdz3_down   =  sympy.diff(prefactor*z*(H-2*z)**(1/a),  z, 3)

sym_dKdz_up     =  sympy.diff(prefactor* (H-z)*(2*z-1)**(1/a), z, 1)
sym_d2Kdz2_up   =  sympy.diff(prefactor* (H-z)*(2*z-1)**(1/a), z, 2)
sym_d3Kdz3_up   =  sympy.diff(prefactor* (H-z)*(2*z-1)**(1/a), z, 3)

dKdz_down   =  sympy.utilities.lambdify(z,          sym_dKdz_down,np)
d2Kdz2_down =  sympy.utilities.lambdify(z,          sym_d2Kdz2_down,np)
d3Kdz3_down =  sympy.utilities.lambdify(z,          sym_d3Kdz3_down,np)

dKdz_up   =  sympy.utilities.lambdify(z,          sym_dKdz_up,np)
d2Kdz2_up =  sympy.utilities.lambdify(z,          sym_d2Kdz2_up,np)
d3Kdz3_up =  sympy.utilities.lambdify(z,          sym_d3Kdz3_up,np)

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

def pycnocline_down(z):
    a = 1
    H = 1
    Kbar = 1
    prefactor = Kbar*2*(1+a)*(1+2*a)/(a**2*H**(1+1/a))
    return prefactor*z*(H-2*z)**(1/a)
K_down=pycnocline_down
def pycnocline_up(z):
    a = 1
    H = 1
    Kbar = 1
    prefactor = Kbar*2*(1+a)*(1+2*a)/(a**2*H**(1+1/a))
    return prefactor*(H-z)*(2*z-1)**(1/a)
K_up=pycnocline_up

def dKdz(z):
    return np.where(z < H/2,  dKdz_down(z), dKdz_up(z))

def alpha(w,z):
    return np.where(z < H/2,  w+dKdz_down(z),w+dKdz_up(z))

def dAdz(w,z):
    return np.where(z < H/2, d2Kdz2_down(z),d2Kdz2_up(z))

def ddAdzz(w,z):
    return np.where(z < H/2, d3Kdz3_down(z),d3Kdz3_up(z))

def beta(z):
    mask=(z<H/2)
    temp=np.empty_like(z);
    prefactor=np.sqrt(2)
    temp[mask]=prefactor*np.sqrt(K_down(z[mask]))
    temp[~mask]=prefactor*np.sqrt(K_up(z[~mask]))
    return temp

def dBdz(z):
    mask=(z<H/2)
    temp=np.empty_like(z);
    prefactor=np.sqrt(2)/2
    temp[mask]= prefactor*(np.power(K_down(z[mask]), -1/2))*dKdz_down(z[mask])
    temp[~mask]=prefactor*(np.power(K_up(z[~mask]), -1/2))*dKdz_up(z[~mask])
    return temp


def ddBdzz(z):
    mask=(z<H/2)
    temp=np.empty_like(z);
    prefactor=np.sqrt(2)/2
   
    temp[mask]=prefactor*(-1/2* np.power(K_down(z[mask]), -3/2)* dKdz_down(z[mask])**2+\
        np.power(K_down(z[mask]),-1/2)*d2Kdz2_down(z[mask]))
    
    temp[~mask]=prefactor*(-1/2* np.power(K_up(z[~mask]), -3/2)* dKdz_up(z[~mask])**2+\
        np.power(K_up(z[~mask]),-1/2)*d2Kdz2_up(z[~mask]))
    return temp
    #return np.where(z < H/2, ddBdzz_down(z),ddBdzz_up(z))

def dABdz(w,z):
    mask=(z<H/2)
    temp=np.empty_like(z);
    temp[mask]=d2Kdz2_down(z[mask])*np.sqrt(2*K_down(z[mask]))+\
            (w+dKdz_down(z[mask]))* np.sqrt(2)/2* np.power(K_down(z[mask]),-1/2) *dKdz_down(z[mask])
            
    temp[~mask]=d2Kdz2_up(z[~mask])*np.sqrt(2*K_up(z[~mask]))+\
            (w+dKdz_up(z[~mask]))* np.sqrt(2)/2* np.power(K_up(z[~mask]),-1/2) *dKdz_up(z[~mask])
            
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
#    return z + alpha(w,z)*dt+beta(z)*dW+\
#            1/2*beta(z)*dBdz(z)*(dW*dW-dt)+\
#            1/2*(dABdz(w,z)+1/2*ddBdzz(z)*beta(z)**2)*dW*dt+\
#            1/2*(alpha(w,z)*dAdz(w,z)+1/2*ddAdzz(w,z)*beta(z)**2)*dt**2

##############
#Milstein 2nd#
##############
def step_v(z,w,dt,N_sample):
    dW=np.random.normal(0,np.sqrt(dt),N_sample)
    temp= z + alpha(w,z)*dt + beta(z + 1/2*alpha(w,z)*dt)*dW
    temp=np.where(temp>1.0, 1-(temp-1),temp)
    temp=np.where(temp<0.0, -temp,temp)
    return temp
