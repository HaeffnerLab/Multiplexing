import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import curve_fit
import math
from scipy import interpolate
from scipy.integrate import solve_ivp
from Eqposition3 import *

E0=8.85*10**-12
e=1.6*10**-19
m=6.64215627*10**(-26)

def ions5_ode(t,y,L,T,mode):
    x1,x2,x3,x4,x5,v1,v2,v3,v4,v5=y
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
    dx3_dt=v3
    dx4_dt=v4
    dx5_dt=v5
    x=[x1,x2,x3,x4,x5]
    x.sort()
    if x[0]==x1:
        dv1_dt=1/m*((-2*a*x1-b)-e**2/(4*math.pi*E0)*(1/(x2-x1)**2+1/(x3-x1)**2+1/(x4-x1)**2+1/(x5-x1)**2))
    elif x[0]==x2:
        dv2_dt=1/m*((-2*a*x2-b)-e**2/(4*math.pi*E0)*(1/(x2-x1)**2+1/(x3-x2)**2+1/(x4-x2)**2+1/(x5-x2)**2))
    elif x[0]==x3:
        dv3_dt=1/m*((-2*a*x3-b)-e**2/(4*math.pi*E0)*(1/(x3-x1)**2+1/(x3-x2)**2+1/(x4-x3)**2+1/(x5-x3)**2))
    elif x[0]==x4:
        dv4_dt=1/m*((-2*a*x4-b)-e**2/(4*math.pi*E0)*(1/(x4-x1)**2+1/(x4-x2)**2+1/(x4-x3)**2+1/(x5-x4)**2))
    elif x[0]==x5:
        dv5_dt=1/m*((-2*a*x4-b)-e**2/(4*math.pi*E0)*(1/(x5-x1)**2+1/(x5-x2)**2+1/(x5-x3)**2+1/(x5-x4)**2))
    if x[1]==x1:
        dv1_dt=1/m*((-2*a*x[1]-b)+e**2/(4*math.pi*E0)*(1/(x[0]-x[1])**2-1/(x[2]-x[1])**2-1/(x[3]-x[1])**2-1/(x[4]-x[1])**2))
    elif x[1]==x2:
        dv2_dt=1/m*((-2*a*x[1]-b)+e**2/(4*math.pi*E0)*(1/(x[0]-x[1])**2-1/(x[2]-x[1])**2-1/(x[3]-x[1])**2-1/(x[4]-x[1])**2))
    elif x[1]==x3:
        dv3_dt=1/m*((-2*a*x[1]-b)+e**2/(4*math.pi*E0)*(1/(x[0]-x[1])**2-1/(x[2]-x[1])**2-1/(x[3]-x[1])**2-1/(x[4]-x[1])**2))
    elif x[1]==x4:
        dv4_dt=1/m*((-2*a*x[1]-b)+e**2/(4*math.pi*E0)*(1/(x[0]-x[1])**2-1/(x[2]-x[1])**2-1/(x[3]-x[1])**2-1/(x[4]-x[1])**2))
    elif x[1]==x5:
        dv5_dt=1/m*((-2*a*x[1]-b)+e**2/(4*math.pi*E0)*(1/(x[0]-x[1])**2-1/(x[2]-x[1])**2-1/(x[3]-x[1])**2-1/(x[4]-x[1])**2))
    if x[2]==x1:
        dv1_dt=1/m*((-2*a*x[2]-b)+e**2/(4*math.pi*E0)*(1/(x[0]-x[2])**2+1/(x[2]-x[1])**2-1/(x[3]-x[2])**2-1/(x[4]-x[2])**2))
    elif x[2]==x2:
        dv2_dt=1/m*((-2*a*x[2]-b)+e**2/(4*math.pi*E0)*(1/(x[0]-x[2])**2+1/(x[2]-x[1])**2-1/(x[3]-x[2])**2-1/(x[4]-x[2])**2))
    elif x[2]==x3:
        dv3_dt=1/m*((-2*a*x[2]-b)+e**2/(4*math.pi*E0)*(1/(x[0]-x[2])**2+1/(x[2]-x[1])**2-1/(x[3]-x[2])**2-1/(x[4]-x[2])**2))
    elif x[2]==x4:
        dv4_dt=1/m*((-2*a*x[2]-b)+e**2/(4*math.pi*E0)*(1/(x[0]-x[2])**2+1/(x[2]-x[1])**2-1/(x[3]-x[2])**2-1/(x[4]-x[2])**2))
    elif x[2]==x5:
        dv5_dt=1/m*((-2*a*x[2]-b)+e**2/(4*math.pi*E0)*(1/(x[0]-x[2])**2+1/(x[2]-x[1])**2-1/(x[3]-x[2])**2-1/(x[4]-x[2])**2))
    if x[3]==x1:
        dv1_dt=1/m*((-2*a*x[3]-b)+e**2/(4*math.pi*E0)*(1/(x[0]-x[3])**2+1/(x[3]-x[1])**2+1/(x[3]-x[2])**2-1/(x[4]-x[3])**2))
    elif x[3]==x2:
        dv2_dt=1/m*((-2*a*x[3]-b)+e**2/(4*math.pi*E0)*(1/(x[0]-x[3])**2+1/(x[3]-x[1])**2+1/(x[3]-x[2])**2-1/(x[4]-x[3])**2))
    elif x[3]==x3:
        dv3_dt=1/m*((-2*a*x[3]-b)+e**2/(4*math.pi*E0)*(1/(x[0]-x[3])**2+1/(x[3]-x[1])**2+1/(x[3]-x[2])**2-1/(x[4]-x[3])**2))
    elif x[3]==x4:
        dv4_dt=1/m*((-2*a*x[3]-b)+e**2/(4*math.pi*E0)*(1/(x[0]-x[3])**2+1/(x[3]-x[1])**2+1/(x[3]-x[2])**2-1/(x[4]-x[3])**2))
    elif x[3]==x5:
        dv5_dt=1/m*((-2*a*x[3]-b)+e**2/(4*math.pi*E0)*(1/(x[0]-x[3])**2+1/(x[3]-x[1])**2+1/(x[3]-x[2])**2-1/(x[4]-x[3])**2))
    if x[4]==x1:
        dv1_dt=1/m*((-2*a*x[4]-b)+e**2/(4*math.pi*E0)*(1/(x[0]-x[4])**2+1/(x[4]-x[1])**2+1/(x[4]-x[2])**2+1/(x[4]-x[3])**2))
    elif x[4]==x2:
        dv2_dt=1/m*((-2*a*x[4]-b)+e**2/(4*math.pi*E0)*(1/(x[0]-x[4])**2+1/(x[4]-x[1])**2+1/(x[4]-x[2])**2+1/(x[4]-x[3])**2))
    elif x[4]==x3:
        dv3_dt=1/m*((-2*a*x[4]-b)+e**2/(4*math.pi*E0)*(1/(x[0]-x[4])**2+1/(x[4]-x[1])**2+1/(x[4]-x[2])**2+1/(x[4]-x[3])**2))
    elif x[4]==x4:
        dv4_dt=1/m*((-2*a*x[4]-b)+e**2/(4*math.pi*E0)*(1/(x[0]-x[4])**2+1/(x[4]-x[1])**2+1/(x[4]-x[2])**2+1/(x[4]-x[3])**2))
    elif x[4]==x5:
        dv5_dt=1/m*((-2*a*x[4]-b)+e**2/(4*math.pi*E0)*(1/(x[0]-x[4])**2+1/(x[4]-x[1])**2+1/(x[4]-x[2])**2+1/(x[4]-x[3])**2))
    return [dx1_dt,dx2_dt,dx3_dt,dx4_dt,dx5_dt,dv1_dt,dv2_dt,dv3_dt,dv4_dt,dv5_dt]

a0=7.93*10**-14
omega=math.sqrt(2*a0/m)
l=math.pow(e**2/(4*math.pi*E0*m*omega**2),1/3)
x1,x2,x3,x4,x5=Eqposition3(5)*l
initial_state=[x1,x2,x3,x4,x5,0,0,0,0,0]
solution=solve_ivp(ions5_ode,(0,4*10**-4),initial_state,t_eval=np.linspace(0,4*10**-4,10000),method='DOP853',args=(8*10**-5,2*10**-4,'linear'))
t=solution.t
x1=solution.y[0]
x2=solution.y[1]
x3=solution.y[2]
x4=solution.y[3]
x5=solution.y[4]
Amplitude_1=0.5*(np.max(x1[5000:])-np.min(x1[5000:]))
Amplitude_2=0.5*(np.max(x2[5000:])-np.min(x2[5000:]))
print(f'the amplitude of the first ion is {Amplitude_1}')
print(f'the amplitude of the second ion is {Amplitude_2}')
plt.plot(10**6*t,10**6*x1,label='the first ion')
plt.plot(10**6*t,10**6*x2,label='the second ion')
plt.plot(10**6*t,10**6*x3,label='the third ion')
plt.plot(10**6*t,10**6*x4,label='the fourth ion')
plt.plot(10**6*t,10**6*x5,label='the fifth ion')
plt.legend()
plt.xlabel('time/us')
plt.ylabel('the positions of the ions')
plt.show()


