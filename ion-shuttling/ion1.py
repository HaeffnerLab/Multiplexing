import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_ivp
import time

def ion1(t,y,L,T,mode):
    x,v=y
    if mode=='flat':
        x0=-L/2*(1-10/9*(math.cos(math.pi*t/T)-0.1*math.cos(math.pi*3*t/T)))
    if mode=='sinusoidal':
        x0=-L/2*(1-math.cos(math.pi*t/T))
    if 0<=t<=T:
        a=-1.85*10**-3*x0**3-3.41*10**-7*x0**2-1.39*10**-11*x0+7.93*10**-14
        b=-3.44*10**-11*x0**2-1.61*10**-13*x0-4.72*10**-20
    else:
        a=-1.85*10**-3*(-L)**3-3.41*10**-7*(-L)**2-1.39*10**-11*(-L)+7.93*10**-14
        b=-3.44*10**-11*(-L)**2-1.61*10**-13*(-L)-4.72*10**-20
    dx_dt=v
    dv_dt=1/(6.64215627*10**(-26))*(-2*a*x-b)
    return [dx_dt,dv_dt]

start_time=time.time()
solution=solve_ivp(ion1,(0,4*10**-4),[0,0],t_eval=np.linspace(0,4*10**-4,10000),max_step=7**-10,atol=7**-16,args=(8*10**-5,2*10**-4,'flat'))
end_time=time.time()
time_elapsed=end_time-start_time
x=solution.y[0]
Amplitude=0.5*(np.max(x[8000:])-np.min(x[8000:]))
quantum=1.628*10**-28
n=7.91768*10**-14*Amplitude**2/quantum
print('mode=flat')
print('T=2*10**-4')
print(f'the amplitude is {Amplitude}')
print(f'quata: {n}')
print(f'calculation time: {time_elapsed}')
