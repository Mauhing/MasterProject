#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 14:23:39 2018

@author: mhyip
"""
import numpy as np
from matplotlib import pyplot as plt

ZE1=np.load("ZE1.npy")
ZE32=np.load("ZE32.npy")
TimeArray=np.load("TimeArray.npy")

fig=plt.figure()
plt.plot(TimeArray,ZE1, label="Euler 1 time step")
plt.plot(TimeArray[0::32],ZE32, label="Euler 32 time steps")
plt.xlabel("Time (s)")
plt.ylabel("Particle position (m)")
plt.legend()
