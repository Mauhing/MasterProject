#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 22:17:54 2018

@author: mhyip
"""

import numpy as np
#import matplotlib
#matplotlib inline
from matplotlib import pyplot as plt
# Get nicer looking plots than default
plt.style.use('bmh')


hist_M2=np.load("M2.dat")
#np.savetxt('M2.txt', hist_M2, fmt='%.10f')

hist_M2_Const=np.load("M2_const.dat")

hist_V=np.load("V.dat")

hist_V_Const=np.load("V_const.dat")

hist_m=np.load("M.dat")

midpoints=np.load("midpoints.dat")

fig=plt.figure(1)
plt.subplot(2,3,1)
plt.plot(hist_V, midpoints, linewidth=0.5, label= "Visser")
plt.plot(hist_V_Const, midpoints, linewidth=0.5, label= "Visser_const")
plt.xlabel('Concetration')
plt.ylabel('Depth /m')
plt.title('Visser')
plt.legend()
plt.gca().invert_yaxis()

plt.subplot(2,3,2)
plt.plot(hist_m, midpoints, linewidth=0.5, label= "Visser")
plt.plot(hist_M2_Const, midpoints, linewidth=0.5, label= "Visser_const")
plt.xlabel('Concetration')
plt.ylabel('Depth /m')
plt.title('Milstein 1 nd')
plt.legend()
plt.gca().invert_yaxis()


plt.subplot(2,3,3)
plt.plot(hist_M2, midpoints, linewidth=0.5, label= "Milstein_2")
plt.plot(hist_M2_Const, midpoints, linewidth=0.5, label= "Milstein_2_const")
plt.xlabel('Concetration')
plt.ylabel('Depth /m')
plt.title('Milstein 2 nd')
plt.legend()
plt.gca().invert_yaxis()

plt.show()