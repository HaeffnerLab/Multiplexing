import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import curve_fit
import math
from scipy import interpolate
from scipy.integrate import solve_ivp

data=np.loadtxt('1.15,0.01,0.0005.csv',delimiter=',',skiprows=9)

def mini(x_data,y_data): #find the minimum point of the potential
    min_index=np.argmin(y_data)
    return x_data[min_index]
a=[]
b=[]
c=[]
def pfit(x_data,y_data): #use second order polynomials to fit the potential
    m=6.64215627*10**(-26)
    degree=2
    fit_x_data=[]
    fit_y_data=[]
    for j in x_data:
        if mini(x_data,y_data)-0.0001<=j and j<=mini(x_data,y_data)+0.0001:
            fit_x_data.append(j)
            fit_y_data.append(y_data[x_data.tolist().index(j)])
    coeffs=np.polyfit(fit_x_data,fit_y_data,degree)
    poly=np.poly1d(coeffs)
    a.append(coeffs[0])
    b.append(coeffs[1])
    c.append(coeffs[2])
    derivative=np.polyder(poly)
    y_fit=poly(x_data)
    mse=mean_squared_error(y_data,y_fit)
    r2=r2_score(y_data,y_fit)
    return math.sqrt((2*coeffs[0])/m) #return the angular frequency

x_data=10**-3*data[:,2]
fre=[]
pos=[]
for i in range(20):
    y_data=1.6*10**-19*data[:,i+3]
    fre.append(pfit(x_data,y_data))
    pos.append(mini(x_data,y_data))

delta=[i for i in range(20)]
delta=np.array(delta)
y = 10**6*np.array(pos)
np.save("delta.npy", delta)
np.save("y.npy", y)
plt.plot(delta, y)
plt.xlabel('delta')
plt.ylabel('minimum point of the potential/µm')
plt.show()

for i in range(20):#plot fitted potential and original potential
    x0=pos[i]
    a=-1.85*10**-3*x0**3-3.41*10**-7*x0**2-1.39*10**-11*x0+7.93*10**-14
    b=-3.44*10**-11*x0**2-1.61*10**-13*x0-4.72*10**-20
    c=4.91*10**-14*x0**2-3.46*10**-18*x0+1.6*10**-20
    plt.plot(10**6*x_data,a*x_data**2+b*x_data+c,color='b')
    plt.plot(10**6*x_data,1.6*10**-19*data[:,3+i],color='r')
plt.plot([],[],color='b',label='fit')
plt.plot([],[],color='r',label='original')
plt.xlabel('position/µm')
plt.ylabel('potential')
plt.legend()
plt.show()

def delta_pos(delta,pos): #fit delta as a function of the minimum position of the potential
    coeffs_2=np.polyfit(pos,delta,2)
    linear=np.poly1d(coeffs_2)
    delta_fit=linear(pos)
    r2=r2_score(delta,delta_fit)
    print(r2)
    print(coeffs_2)
    plt.plot(pos,delta,color='r',label='original')
    plt.plot(pos,delta_fit,color='b',label='predicted')
    plt.xlabel('position/m')
    plt.ylabel('$delta$')
    plt.legend()
    plt.show()
    return coeffs_2
delta_pos(delta,pos)


def fre_pos(pos,fre): #to get the second coefficent b as a function of x0
    coeffs_3=np.polyfit(pos,fre,2)
    poly_2=np.poly1d(coeffs_3)
    fre_fit=poly_2(pos)
    r2=r2_score(fre,fre_fit)
    print(r2)
    print(coeffs_3)
    plt.plot(pos,fre,color='r',label='original')
    plt.plot(pos,fre_fit,color='b',label='predicted')
    plt.xlabel('position/m')
    plt.ylabel('b')
    plt.legend()
    plt.show()
    return

fre_pos(pos,b)

def c_fit(pos,c): #to gey the constant term as a function of x0
    coeffs_4=np.polyfit(pos,c,2)
    poly_3=np.poly1d(coeffs_4)
    c_fit=poly_3(pos)
    r2=r2_score(c,c_fit)
    print(r2)
    print(coeffs_4)
    plt.plot(pos,c,color='r',label='original')
    plt.plot(pos,c_fit,color='b',label='predicted')
    plt.xlabel('position/m')
    plt.ylabel('constant coefficient')
    plt.legend()
    plt.show()
    return
c_fit(pos,c)