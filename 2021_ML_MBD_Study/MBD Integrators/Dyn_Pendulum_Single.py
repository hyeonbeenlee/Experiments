import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Functions as F

def dY(t,Y):
    global m,g,L,c
    th=Y[0];    dth=Y[1];

    dY=np.zeros(2)
    dY[0]=dth
    dY[1]=-c/(m*L)*dth-g/L*np.sin(th)

    return dY

## Input Variables
L=0.1 #[0.1,0.2] 0.01
c=0 #[0,0.15] 0.01
th_0=-np.pi/2 #[-pi/2,pi/2], pi/10
# t=[0,2], 200 steps sampled

## Constants
g=9.81;     m=0.3;      dth_0=np.pi/2;

## Output = th,dth,ddth

## State Vector
Y=np.zeros(2)
Y[0]=th_0
Y[1]=dth_0

## Simul Configs
t=0
endTime=2
steps=200
h=endTime/steps
times=np.array([])

QiLog=np.zeros(steps+1)
dQiLog=np.zeros(steps+1)
ddQiLog=np.zeros(steps+1)

for i in range(steps+1):
    # print(f"Solving t={t:.5f}(sec)")
    k1=dY(t,Y)
    k2=dY(t+0.5*h,Y+k1*0.5*h)
    k3=dY(t+0.5*h,Y+k2*0.5*h)
    k4=dY(t+h,Y+k3*h)
    Grad=(k1+2*k2+2*k3+k4)/6
    Y_Next=Y+Grad*h

    QiLog[i]=Y_Next[0]
    dQiLog[i]=Y_Next[1]
    ddQiLog[i]=Grad[1]

    Y=Y_Next
    times=np.append(times,t)
    t=t+h

QiIdx=['th']
dQiIdx=['dth']
ddQiIdx=['ddth']
Times=pd.DataFrame(times,columns=['Time']).T
QiDF=pd.DataFrame(QiLog,columns=QiIdx).T
dQiDF=pd.DataFrame(dQiLog,columns=dQiIdx).T
ddQiDF=pd.DataFrame(ddQiLog,columns=ddQiIdx).T
Result=pd.concat([Times,QiDF,dQiDF,ddQiDF])
Result.loc['L']=L
Result.loc['c']=c
Result.loc['th_0']=th_0
Result=Result.T
print(Result)
# Result.to_csv(f"L={L:.3f},c={c:.3f},th_0={th_0:.3f}.csv",header=False)



# Idx=['Time']+QiIdx+dQiIdx+ddQiIdx
# Sampled_Result=pd.DataFrame(0,index=Idx,columns=range(len(Times)))
# for i in range(len(Result.loc['Time'])):
#     # if i%50==0:
#     Sampled_Result[i]=Result[i]


# print(f'L={L}, c={c}, th_0={th_0}')
# print(Sampled_Result)
# Sampled_Result.to_csv(f'./MBD Data/Pendulum_Single/L={L:.2f},c={c:.2f},th_0={pinums-5:d}pi_10.csv',header=False)
# print(f"L={L:.2f},c={c:.2f},th_0={pinums-5:d}pi/10 saved.")




## 플로팅용 코드
plt.suptitle(f'L={L}, c={c}, $\\theta^0$={th_0}')
plt.subplot(311)
plt.plot(times,QiLog)
plt.grid()

plt.subplot(312)
plt.plot(times,dQiLog)
plt.grid()

plt.subplot(313)
plt.plot(times,ddQiLog)
plt.grid()

plt.tight_layout()
plt.show()