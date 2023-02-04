import numpy as np
import Functions as F
import pandas as pd
import matplotlib.pyplot as plt

def FB_Drive(t):
    global dth2
    th2=dth2*t
    return th2

def FB_C(Q,t):
    global u1o,u2o,u2a,u3a,u3b,u4b,u4c,u1c,th2_0
    Rx1 = Q[0];     Ry1 = Q[1];     th1 = Q[2]
    Rx2 = Q[3];     Ry2 = Q[4];     th2 = Q[5]
    Rx3 = Q[6];     Ry3 = Q[7];     th3 = Q[8]
    Rx4 = Q[9];     Ry4 = Q[10];    th4 = Q[11]

    C = np.zeros(12)
    C[0:3]=[Rx1,Ry1,th1]
    C[3:5]=(np.array([Rx1,Ry1])+F.A(th1)@u1o)-(np.array([Rx2,Ry2])+F.A(th2)@u2o) #Rev O
    C[5:7]=(np.array([Rx2,Ry2])+F.A(th2)@u2a)-(np.array([Rx3,Ry3])+F.A(th3)@u3a) #Rev A
    C[7:9]=(np.array([Rx3,Ry3])+F.A(th3)@u3b)-(np.array([Rx4,Ry4])+F.A(th4)@u4b) #Rev B
    C[9:11]=(np.array([Rx1,Ry1])+F.A(th1)@u1c)-(np.array([Rx4,Ry4])+F.A(th4)@u4c) #Rev C
    C[11] = th2-th2_0-FB_Drive(t) #Driving Constraint
    return C

def FB_Cq(Q):
    global u1o,u2o,u2a,u3a,u3b,u4b,u4c,u1c
    Rx1 = Q[0];     Ry1 = Q[1];     th1 = Q[2];
    Rx2 = Q[3];     Ry2 = Q[4];     th2 = Q[5];
    Rx3 = Q[6];     Ry3 = Q[7];     th3 = Q[8];
    Rx4 = Q[9];     Ry4 = Q[10];    th4 = Q[11];

    Cq=np.zeros((12,12))
    for i in range(3):
        Cq[i,i]=1 #Ground
    Cq[3:5,0:6]=np.hstack([np.eye(2),(F.At(th1)@u1o).reshape(2,1),-np.eye(2),(-F.At(th2)@u2o).reshape(2,1)]) #R1-R2,RevO
    Cq[5:7,3:9]=np.hstack([np.eye(2),(F.At(th2)@u2a).reshape(2,1),-np.eye(2),(-F.At(th3)@u3a).reshape(2,1)]) #R2-R3,RevA
    Cq[7:9,6:12]=np.hstack([np.eye(2),(F.At(th3)@u3b).reshape(2,1),-np.eye(2),(-F.At(th4)@u4b).reshape(2,1)]) #R3-R4,RevB
    Cq[9:11,0:3]=np.hstack([np.eye(2),(F.At(th1)@u1c).reshape(2,1)])
    Cq[9:11,9:12]=np.hstack([-np.eye(2),(-F.At(th4)@u4c).reshape(2,1)]) #R1-R4,RevC
    Cq[11,5]=1 #Driving Constraint
    return Cq

def FB_Qd(Q,dQ):
    global l1,l2,l3,l4
    Rx1 = Q[0];     Ry1 = Q[1];     th1 = Q[2];
    Rx2 = Q[3];     Ry2 = Q[4];     th2 = Q[5];
    Rx3 = Q[6];     Ry3 = Q[7];     th3 = Q[8];
    Rx4 = Q[9];     Ry4 = Q[10];    th4 = Q[11];
    dth1=dQ[2];     dth2=dQ[5];     dth3=dQ[8];     dth4=dQ[11];
    
    Qd=np.zeros((12,1))
    Qd[5]=dth2**2*l2*np.cos(th2)
    Qd[6]=dth2**2*l2*np.sin(th2)
    Qd[7]=dth3**2*l3*np.cos(th3)
    Qd[8]=dth3**2*l3*np.sin(th3)
    Qd[9]=l1*np.cos(th1)*dth1**2 - l4*np.cos(th4)*dth4**2
    Qd[10]=l1*np.sin(th1)*dth1**2 - l4*np.sin(th4)*dth4**2
    return Qd

def FB_Ct():
    global dth2
    Ct=np.zeros((12,1))
    Ct[-1]=-dth2
    return Ct

def FB_Cqt():
    Cqt=np.zeros((12,12))
    return Cqt

def FB_Ctt():
    Ctt=np.zeros((12,1))
    return Ctt

# Loop Closure Eqn for th3 th4 solve
def LC(th3,th4):
    global l1,l2,l3,l4,th2_0
    Eqn1=l2*np.cos(th2_0)+l3*np.cos(th3)+l4*np.cos(th4)-l1 #==0
    Eqn2=l2*np.sin(th2_0)+l3*np.sin(th3)+l4*np.sin(th4) #==0
    Eqn=np.array([Eqn1,Eqn2])
    return Eqn
# Loop Closure Jacobian
def LCJ(th3,th4):
    global l1,l2,l3,l4,th2_0
    J=np.zeros((2,2))
    J[0,:]=[-l3*np.sin(th3),-l4*np.sin(th4)]
    J[1,:]=[l3*np.cos(th3),l4*np.cos(th4)]
    return J



## Constants
l1=0.35
l2=0.2
l3=0.35
l4=0.25

## Initial Conditions
dth2=5
th2_0=np.deg2rad(57.27)







## Solve initial th3 th4
# th3 th4 초기 추측값
th3_0=np.deg2rad(0);      th4_0=np.deg2rad(200);

# NR Solve
for i in range(100):
    Solution=np.array([th3_0,th4_0])-np.linalg.inv(LCJ(th3_0,th4_0))@LC(th3_0,th4_0)
    th3_0=Solution[0]
    th4_0=Solution[1]
    if np.linalg.norm(Solution)<=1e-6:
        break

print(np.rad2deg(Solution))
Go=input(f'Initial th3(deg) and th4(deg) are solved for th2={np.rad2deg(th2_0)}(deg). Proceed? : ')
if Go=='y' or Go=='Y':
    print('Simulation starts.')
else:
    quit()







## Local Coordinates
u1o=np.array([0,0]);    u2o=np.array([0,0]); #O
u2a=np.array([l2,0]);   u3a=np.array([0,0]); #A
u3b=np.array([l3,0]);   u4b=np.array([0,0]); #B
u4c=np.array([l4,0]);   u1c=np.array([l1,0]); #C

## Initial Position
Rx1=0; Ry1=0; th1=0;
Rx2=0; Ry2=0; th2=th2_0;
Rx3=l2*np.cos(th2); Ry3=l2*np.sin(th2); th3=th3_0;
Rx4=Rx3+l3*np.cos(th3); Ry4=Ry3+l3*np.sin(th3); th4=th4_0;
Q=np.array([Rx1,Ry1,th1,Rx2,Ry2,th2,Rx3,Ry3,th3,Rx4,Ry4,th4]).T


## Simulation Configs
t=0
endTime=1
steps=1000;
h=endTime/steps
etol=1e-5 # For NR Solve
times=np.array([])

## Components
C=FB_C(Q,t)
Cq=FB_Cq(Q)
Ct=FB_Ct()
Cqt=FB_Cqt()
Ctt=FB_Ctt()

## For Record
QiLog=np.zeros((len(Q),steps+1))
dQiLog=np.zeros((len(Q),steps+1))
ddQiLog=np.zeros((len(Q),steps+1))


for i in range(steps+1):
    print(f"Solving t={t:.5f}(sec)")
    ## Newton Raphson Position Solve
    while True:
        Qvar=np.linalg.solve(Cq,-C)
        Q=Q+Qvar
        C=FB_C(Q,t)
        Cq=FB_Cq(Q)
        if np.linalg.norm(Qvar)<=etol or np.linalg.norm(C)<=etol:
            break
    Record_Q=Q.T


    ## Velocity
    Cq=FB_Cq(Q)
    Ct=FB_Ct()
    dQ=np.linalg.solve(Cq,-Ct)
    Record_dQ=dQ.T

    ## Acceleration
    Qd=FB_Qd(Q,dQ)
    ddQ=np.linalg.solve(Cq,Qd)
    Record_ddQ=ddQ.T

    ## Record
    QiLog[:,i]=Record_Q
    dQiLog[:,i]=Record_dQ
    ddQiLog[:,i]=Record_ddQ

    times = np.append(times,t)
    t=t+h

QiIdx=['Rx1','Ry1','th1','Rx2','Ry2','th2','Rx3','Ry3','th3','Rx4','Ry4','th4']
dQiIdx=['dRx1','dRy1','dth1','dRx2','dRy2','dth2','dRx3','dRy3','dth3','dRx4','dRy4','dth4']
ddQiIdx=['ddRx1','ddRy1','ddth1','ddRx2','ddRy2','ddth2','ddRx3','ddRy3','ddth3','ddRx4','ddRy4','ddth4']
Times=pd.DataFrame(times,columns=['Time']).T

QiDF=pd.DataFrame(QiLog,index=QiIdx)
dQiDF=pd.DataFrame(dQiLog,index=dQiIdx)
ddQiDF=pd.DataFrame(ddQiLog,index=ddQiIdx)

Result=pd.concat([Times,QiDF,dQiDF,ddQiDF])
Result=Result.T
print(Result)







plt.subplot(331)
plt.plot(Result['Time'],Result['th2'])
plt.grid()

plt.subplot(332)
plt.plot(Result['Time'],Result['th3'])
plt.grid()

plt.subplot(333)
plt.plot(Result['Time'],Result['th4'])
plt.grid()

plt.subplot(334)
plt.plot(Result['Time'],Result['dth2'])
plt.grid()

plt.subplot(335)
plt.plot(Result['Time'],Result['dth3'])
plt.grid()

plt.subplot(336)
plt.plot(Result['Time'],Result['dth4'])
plt.grid()

plt.subplot(337)
plt.plot(Result['Time'],Result['ddth2'])
plt.grid()

plt.subplot(338)
plt.plot(Result['Time'],Result['ddth3'])
plt.grid()

plt.subplot(339)
plt.plot(Result['Time'],Result['ddth4'])
plt.grid()

plt.tight_layout()
plt.show()



