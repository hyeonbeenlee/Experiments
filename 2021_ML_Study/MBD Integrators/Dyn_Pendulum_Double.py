import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Functions as F

def dY(t,Y):
    global m1,m2,g,l1,l2
    th1=Y[0];   th2=Y[1];
    dth1=Y[2];  dth2=Y[3];

    dY = np.zeros(4)
    dY[0] = dth1
    dY[1] = dth2

    # MIT Solution
    # ddth1=(-g*(2*m1+m2)*np.sin(th1)-m2*g*np.sin(th1-2*th2)-2*np.sin(th1-th2)*m2*(dth2**2*l2+dth1**2*l1*np.cos(th1-th2)))
    # ddth1=ddth1/(l1*(2*m1+m2-m2*np.cos(2*th1-2*th2)))
    # ddth2=(2*np.sin(th1-th2)*(dth1**2*l1*(m1+m2)+(m1+m2)*g*np.cos(th1)+dth2**2*m2*l2*np.cos(th1-th2)))
    # ddth2=ddth2/(l2*(2*m1+m2-m2*np.cos(2*th1-2*th2)))
    # dY[2]=ddth1
    # dY[3]=ddth2




    Mi=np.zeros((2,2))
    Mi[0,:]=[(m1+m2)*l1,m2*l2*np.cos(th1-th2)]
    Mi[1,:]=[m2*l1*np.cos(th1-th2),m2*l2]

    Qi=np.zeros(2)
    Qi[0]=-(dth2**2)*m2*l2*np.sin(th1-th2)-(m1+m2)*g*np.sin(th1)
    Qi[1]=(dth1**2)*m2*l1*np.sin(th1-th2)-m2*g*np.sin(th2)

    dY[2:4]=np.linalg.solve(Mi,Qi)
    return dY

## Constants
g=9.81;     m1=2;      m2=1;
th1_0=1.6;  th2_0=1.6


## Input Variables
l1=1.473 #[1,2] 0.1
l2=2.829 #[2,3] 0.1
dth1_0=0.092 #[0,0.1] 0.01
dth2_0=0.327 #[0.3,0.5] 0.02
# t=[0,5], 500 steps sampled

## Output = th1,th2,dth1,dth2


## State Vector
Y=np.zeros(4)
Y =[th1_0,th2_0,dth1_0,dth2_0]

## Simul Configs
t=0
endTime=5
steps=500
h=endTime/steps
times=np.array([])

QiLog=np.zeros((2,steps+1))
dQiLog=np.zeros((2,steps+1))
ddQiLog=np.zeros((2,steps+1))

for i in range(steps+1):
    print(f"Solving t={t:.5f}(sec)")
    k1=dY(t,Y)
    k2=dY(t+0.5*h,Y+k1*0.5*h)
    k3=dY(t+0.5*h,Y+k2*0.5*h)
    k4=dY(t+h,Y+k3*h)
    Grad=(k1+2*k2+2*k3+k4)/6
    Y_Next=Y+Grad*h

    QiLog[:,i]=Y_Next[0:2]
    dQiLog[:,i]=Y_Next[2:5]
    ddQiLog[:,i]=k1[2:5]

    Y=Y_Next
    times=np.append(times,t)
    t=t+h

QiIdx=['th1','th2']
dQiIdx=['dth1','dth2']
ddQiIdx=['ddth1','ddth2']
Times=pd.DataFrame(times,columns=['Time']).T
QiDF=pd.DataFrame(QiLog,index=QiIdx)
dQiDF=pd.DataFrame(dQiLog,index=dQiIdx)
ddQiDF=pd.DataFrame(ddQiLog,index=ddQiIdx)
Result=pd.concat([Times,QiDF,dQiDF,ddQiDF])
Result.loc['L1']=l1
Result.loc['L2']=l2
Result.loc['dth1_0']=dth1_0
Result.loc['dth2_0']=dth2_0
Result=Result.T
print(Result)

# Result.to_csv(f"L1={l1:.3f},L2={l2:.3f},dth1_0={dth1_0:.3f},dth2_0={dth2_0:.3f}.csv",header=False)




# Idx=['Time']+QiIdx+dQiIdx+ddQiIdx
# Sampled_Result=pd.DataFrame(0,index=Idx,columns=range(len(Times)))
# for i in range(len(Result.loc['Time'])):
#     # if i%20==0:
#     Sampled_Result[i]=Result[i]
#
# print(Sampled_Result)



Title1="Double Pendulum\n"
Title2=f"L1={l1:.1f}, L2={l2:.1f}, $\\dot\\theta_1^0$={dth1_0}, $\\dot\\theta_2^0$={dth2_0}"
plt.suptitle(Title1+Title2)

plt.subplot(321)
plt.plot(Result['Time'],Result['th1'])
plt.grid()

plt.subplot(322)
plt.plot(Result['Time'],Result['th2'])
plt.grid()

plt.subplot(323)
plt.plot(Result['Time'],Result['dth1'])
plt.grid()

plt.subplot(324)
plt.plot(Result['Time'],Result['dth2'])
plt.grid()

plt.subplot(325)
plt.plot(Result['Time'],Result['ddth1'])
plt.grid()

plt.subplot(326)
plt.plot(Result['Time'],Result['ddth2'])
plt.grid()

plt.tight_layout()
plt.show()