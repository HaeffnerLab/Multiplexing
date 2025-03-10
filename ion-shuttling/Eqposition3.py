from scipy.optimize import fsolve, leastsq,root,brentq,newton,curve_fit
import math
import timeit
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from time import time 


def Eqposition3(N):
    t0 = time()
    
    def func(u): #the equations
        p=np.empty(shape=(N))
        for m in range(N):
            p[m]=u[m]-np.sum(1/(u[m]-u[:m])**2)+np.sum(1/(u[m]-u[m+1:])**2)
        return p
    
    def dfunc(u): # first order derivative
        p=np.empty(shape=(N))
        for m in range(N):
            p[m]=1+2*np.sum(1/(u[m]-u[:m])**3)-2*np.sum(1/(u[m]-u[m+1:])**3)
        return p
    
    def ddfunc(u): # second order derivative
        p=np.empty(shape=(N))
        for m in range(N):
            p[m]=np.sum(1/(u[m]-u[:m])**4) - np.sum(1/(u[m]-u[m+1:])**4)
        return p
    
    ni = np.arange(0,N)
    guess = 3.94*(N**0.387)*np.sin(1/3*np.arcsin(1.75*N**(-0.982)*((ni+1)-(N+1)/2)))
    
    x0=newton(func,guess,fprime=dfunc,maxiter=100000) # newton method
    #print('Time: ',time()-t0)
    #print(func(x0))

    return(np.round(x0,5))

#print(Eqposition3(2))

