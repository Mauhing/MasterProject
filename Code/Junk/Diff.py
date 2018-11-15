# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 17:35:21 2018

@author: mauhi
"""

import sympy

sympy.init_printing(use_unicode=True)
#%%
H,a,z=sympy.symbols('H a z')

out=sympy.diff(z*(H-2*z)**(1/a),z,1)
print("Lower half 1 ", out,'\n')

out=sympy.diff(z*(H-2*z)**(1/a),z,2)
print("Lower half 2 ", out,'\n')

out=sympy.diff(z*(H-2*z)**(1/a),z,3)
print("Lower half 3 ", out,'\n')

print("-------------------------------------------------")

out=sympy.diff((H-z)*(2*z-1)**(1/a),z,1)
print("Higher half 1 ", out,'\n')

out=sympy.diff((H-z)*(2*z-1)**(1/a),z,2)
print("Higher half 2 ", out,'\n')

out=sympy.diff((H-z)*(2*z-1)**(1/a),z,3)
print("Higher half 3 ", out,'\n')

print("=================================================")

out=sympy.diff(sympy.sqrt(z*(H-2*z)**(1/a)),z,1)
print("Lower half 1 ", out,'\n')

out=sympy.diff(sympy.sqrt(z*(H-2*z)**(1/a)),z,2)
print("Lower half 2 ", out,'\n')

out=sympy.diff(sympy.sqrt(z*(H-2*z)**(1/a)),z,3)
print("Lower half 3 ", out,'\n')

print("-------------------------------------------------")

out=sympy.diff(sympy.sqrt(H-z)*(2*z-1)**(1/a),z,1)
print("Higher half 1 ", out,'\n')

out=sympy.diff(sympy.sqrt(H-z)*(2*z-1)**(1/a),z,2)
print("Higher half 2 ", out,'\n')

out=sympy.diff(sympy.sqrt(H-z)*(2*z-1)**(1/a),z,3)
print("Higher half 3 ", out,'\n')

sympy.init_printing() 

































