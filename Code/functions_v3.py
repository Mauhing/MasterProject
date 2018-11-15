
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
K=pycnocline

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

def d2Kdz(z):
    return np.where(z < H/2, d2Kdz2_down(z),d2Kdz2_up(z))

def d3Kdz(z):
    return np.where(z < H/2, d3Kdz3_down(z),d3Kdz3_up(z))

#%%
#Calculate nest step

#######
#Euler#
#######
def step_e(z,w,dt,N_sample,dW=None):
    if dW is None:
        dW=np.random.normal(0,np.sqrt(dt),N_sample)
    
    a=w+dKdz(z)
    b=np.sqrt(2*K(z))
    
    temp=z+a*dt+b*dW
    temp=np.where(temp>1.0, 1-(temp-1),temp)
    temp=np.where(temp<0.0, -temp,temp)
    return temp

##########
#Milstein#
##########
def step_m(z,w,dt,N_sample,dW=None):
    if dW is None:
        dW=np.random.normal(0,np.sqrt(dt),N_sample)
    
 
    dkdz=dKdz(z)
    b=np.sqrt(2*K(z))
    
    temp= z + w*dt + (1/2)*dkdz*(dW*dW+dt) + b*dW
    temp=np.where(temp>1.0, 1-(temp-1),temp)
    temp=np.where(temp<0.0, -temp,temp)
    return temp


########
#Visser#
########
def step_v(z,w,dt,N_sample,dW=None):
    if dW is None:
        dW=np.random.normal(0,np.sqrt(dt),N_sample)
    
    
    a=w+dKdz(z)
        
    b=np.sqrt(2*K(z + 1/2*a*dt))
    temp= z + a*dt + b*dW
    temp=np.where(temp>1.0, 1-(temp-1),temp)
    temp=np.where(temp<0.0, -temp,temp)
    return temp    


##############
#Milstein 2nd#
##############

def step_m2(z,w,dt,N_sample, dW=None):
    if dW is None:
        dW=np.random.normal(0,np.sqrt(dt),N_sample)
    
    
    
    k=pycnocline(z)
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
    
    temp=np.where(temp>1.0, 1-(temp-1),temp)
    temp=np.where(temp<0.0, -temp,temp)
    return temp

