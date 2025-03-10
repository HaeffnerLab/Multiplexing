import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import curve_fit
import math
from scipy import interpolate
from scipy.integrate import solve_ivp
from Eqposition3 import *
import time

start_time=time.time()
E0=8.85*10**-12
e=1.6*10**-19
m=6.64215627*10**(-26)
def ions2_ode(t,y,L,T,mode):
    x1,x2,v1,v2=y
    if mode=='linear':
        x0=-L/T*t
    if mode=='sinusoidal':
        x0=-L/2*(1-math.cos(math.pi*t/T))
    if 0<=t<=T:
        a=-1.85*10**-3*x0**3-3.41*10**-7*x0**2-1.39*10**-11*x0+7.93*10**-14
        b=-3.44*10**-11*x0**2-1.61*10**-13*x0-4.72*10**-20
    else:
        a=-1.85*10**-3*(-8*10**-5)**3-3.41*10**-7*(-8*10**-5)**2-1.39*10**-11*(-8*10**-5)+7.93*10**-14
        b=-3.44*10**-11*(-8*10**-5)**2-1.61*10**-13*(-8*10**-5)-4.72*10**-20
    
    dx1_dt=v1
    dx2_dt=v2
    if x1>x2:
        dv1_dt=1/m*((-2*a*x1-b)+e**2/(4*math.pi*E0*(x2-x1)**2))
        dv2_dt=1/m*((-2*a*x2-b)-e**2/(4*math.pi*E0*(x2-x1)**2))
    else:
        dv1_dt=1/m*((-2*a*x1-b)-e**2/(4*math.pi*E0*(x2-x1)**2))
        dv2_dt=1/m*((-2*a*x2-b)+e**2/(4*math.pi*E0*(x2-x1)**2))
    return [dx1_dt,dx2_dt,dv1_dt,dv2_dt]

a0=7.93*10**-14
omega=math.sqrt(2*a0/m)
l=math.pow(e**2/(4*math.pi*E0*m*omega**2),1/3)
x1,x2=Eqposition3(2)*l
initial_state=[x1,x2,0,0]
v=[0.05]#the speeds you want to move the ions
a=[]
for i in v:
    transport_time=round(8*10**-5/i,7)
    solution=solve_ivp(ions2_ode,(0,round(4*transport_time/3,7)),initial_state,t_eval=np.linspace(0,round(4*transport_time/3,7),12000),method='RK45',max_step=10**-10,atol=10**-16,args=(8*10**-5,transport_time,'sinusoidal'))
    t=solution.t
    x1=solution.y[0]
    x2=solution.y[1]
    index=9000
    Amplitude_1=0.5*(np.max(x1[index:])-np.min(x1[index:]))
    Amplitude_2=0.5*(np.max(x2[index:])-np.min(x2[index:]))
    print(f'speed={i} m/s')
    print(f'the amplitude of the first ion is {Amplitude_1}')
    print(f'the amplitude of the second ion is {Amplitude_2}')
    end_time=time.time()
    elapsed_time=end_time-start_time
    print(f'Calculation time: {elapsed_time:.6f} seconds')
    a.append(Amplitude_1)

'''
plt.plot(v,a)
plt.xlabel('the speed of moving the potential/(m/s)')
plt.ylabel('the amplitude of the ions/nm')
plt.show()

plt.plot(10**6*t,10**6*x1,label='the first ion')
plt.plot(10**6*t,10**6*x2,label='the second ion')
plt.legend()
plt.xlabel('time/us')
plt.ylabel('the positions of the ions')
plt.show()
'''

