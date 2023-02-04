import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def th(t,tau):
    return t/tau - np.sin(t*tau)/tau**2
def dth(t,tau):
    return 1/tau - np.cos(t*tau)/tau
def ddth(t,tau):
    return np.sin(t*tau)
def pi(t,tau,Lr,r):
    L=Lr*r
    return -np.arcsin((r*np.sin(t/tau - np.sin(t*tau)/tau**2))/L)
def xb(t,tau,Lr,r):
    L=Lr*r
    return L*(1 - (r**2*np.sin(t/tau - np.sin(t*tau)/tau**2)**2)/L**2)**(1/2) + r*np.cos(t/tau - np.sin(t*tau)/tau**2)
#Symbolic math by MATLAB
def dpi(t,tau,Lr,r):
    L=Lr*r
    return -(r*np.cos(t/tau - np.sin(t*tau)/tau**2)*(1/tau - np.cos(t*tau)/tau))/(L*(-(r**2*np.sin((np.sin(t*tau) - t*tau)/tau**2)**2 - L**2)/L**2)**(1/2))
def dxb(t,tau,Lr,r):
    L=Lr*r
    return (r**2*np.sin((np.sin(t*tau) - t*tau)/tau**2)*np.cos(t/tau - np.sin(t*tau)/tau**2)*(1/tau - np.cos(t*tau)/tau))/(L*(-(r**2*np.sin((np.sin(t*tau) - t*tau)/tau**2)**2 - L**2)/L**2)**(1/2)) - r*np.sin(t/tau - np.sin(t*tau)/tau**2)*(1/tau - np.cos(t*tau)/tau)
def ddpi(t,tau,Lr,r):
    L=Lr*r
    return (r*np.sin(t/tau - np.sin(t*tau)/tau**2)*(1/tau - np.cos(t*tau)/tau)**2 - r*np.cos(t/tau - np.sin(t*tau)/tau**2)*np.sin(t*tau) + (dpi(t,tau,Lr,r)*r**2*np.cos(t/tau - np.sin(t*tau)/tau**2)*np.sin(t/tau - np.sin(t*tau)/tau**2)*(1/tau - np.cos(t*tau)/tau))/(L*(-(r**2*np.sin((np.sin(t*tau) - t*tau)/tau**2)**2 - L**2)/L**2)**(1/2)))/(L*(-(r**2*np.sin((np.sin(t*tau) - t*tau)/tau**2)**2 - L**2)/L**2)**(1/2))
def ddxb(t,tau,Lr,r):
    L = Lr * r
    return (dpi(t,tau,Lr,r)*r*np.cos(t/tau - np.sin(t*tau)/tau**2)*(1 - (r**2*np.sin(t/tau - np.sin(t*tau)/tau**2)**2)/L**2)**(1/2)*(1/tau - np.cos(t*tau)/tau))/(-(r**2*np.sin((np.sin(t*tau) - t*tau)/tau**2)**2 - L**2)/L**2)**(1/2) - r*np.sin(t/tau - np.sin(t*tau)/tau**2)*np.sin(t*tau) - (r*np.sin((np.sin(t*tau) - t*tau)/tau**2)*(r*np.sin(t/tau - np.sin(t*tau)/tau**2)*(1/tau - np.cos(t*tau)/tau)**2 - r*np.cos(t/tau - np.sin(t*tau)/tau**2)*np.sin(t*tau) + (dpi(t,tau,Lr,r)*r**2*np.cos(t/tau - np.sin(t*tau)/tau**2)*np.sin(t/tau - np.sin(t*tau)/tau**2)*(1/tau - np.cos(t*tau)/tau))/(L*(-(r**2*np.sin((np.sin(t*tau) - t*tau)/tau**2)**2 - L**2)/L**2)**(1/2))))/(L*(-(r**2*np.sin((np.sin(t*tau) - t*tau)/tau**2)**2 - L**2)/L**2)**(1/2)) - r*np.cos(t/tau - np.sin(t*tau)/tau**2)*(1/tau - np.cos(t*tau)/tau)**2




Input=['Time','tau','r','L/r','th','pi','dpi','ddpi','xb','dxb','ddxb']
ts=np.linspace(0,5,501)

taus=np.linspace(1,2,11) #11 tau steps
rs=np.linspace(1,3,11) #11 r steps
Lrs=np.linspace(2.5,3.5,11) #11 L/r steps

tau=1.780
r=1.360
Lr=3.050


Data=np.zeros((501,11))
for i,t in enumerate(ts):
    _th=th(t,tau)
    _pi=pi(t,tau,Lr,r)
    _dpi=dpi(t,tau,Lr,r)
    _ddpi=ddpi(t,tau,Lr,r)
    _xb=xb(t,tau,Lr,r)
    _dxb=dxb(t,tau,Lr,r)
    _ddxb=ddxb(t,tau,Lr,r)
    Data[i,:]=np.array([t,tau,r,Lr,_th,_pi,_dpi,_ddpi,_xb,_dxb,_ddxb])
Data=pd.DataFrame(Data,columns=Input)



plt.subplot(331)
plt.plot(Data['Time'],Data['xb'])
plt.grid()
plt.subplot(334)
plt.plot(Data['Time'],Data['dxb'])
plt.grid()
plt.subplot(337)
plt.plot(Data['Time'],Data['ddxb'])
plt.grid()
plt.subplot(232)
plt.plot(Data['pi'],Data['dpi'])
plt.grid()
plt.subplot(235)
plt.plot(Data['pi'],Data['ddpi'])
plt.grid()
plt.subplot(233)
plt.plot(Data['xb'],Data['dxb'])
plt.grid()
plt.subplot(236)
plt.plot(Data['xb'],Data['ddxb'])
plt.grid()


plt.show()

