import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import curve_fit
import math


data=np.loadtxt('300V.csv',delimiter=',',skiprows=9)#load comsol data

def mini(x_data,y_data):#find the minimum point of the potential
    min_index=np.argmin(y_data)
    return x_data[min_index]

def pfit(x_data,y_data):
    m=6.64215627*10**(-26)
    degree=2
    fit_x_data=[]
    fit_y_data=[]
    for j in x_data:
        if mini(x_data,y_data)-0.0001<=j and j<=mini(x_data,y_data)+0.0001:
            fit_x_data.append(j)
            fit_y_data.append(y_data[x_data.tolist().index(j)])#choose the data for fitting
    coeffs=np.polyfit(fit_x_data,fit_y_data,degree)
    poly=np.poly1d(coeffs)
    y_fit=poly(x_data)
    mse=mean_squared_error(y_data,y_fit)
    r2=r2_score(y_data,y_fit)
    return math.sqrt((2*coeffs[0])/m)#return trap frequency

x_data=10**-3*data[:,2]#turn the unit into m
fre=[]
pos=[]
for i in range(31):
    y_data=1.6*10**-19*data[:,i+3]
    fre.append(pfit(x_data,y_data))
    pos.append(mini(x_data,y_data))


delta=[i for i in range(31)]#plot the result, may vary depending on data used
delta=np.array(delta)
delta=10*delta
plt.plot(delta,10**6*np.array(fre))
plt.xlabel('voltage')
plt.ylabel('trap frequency')
plt.show()
