import Functions as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def SC_Cq(Q_i):
    global l2,l3,N_Constr,N_Coords
    th2=Q_i[2];     th3=Q_i[5]
    Cq=np.zeros((N_Constr,N_Coords))
    Cq[0,:]=[1,0,l2/2*np.sin(th2),0,0,0,0,0,0]
    Cq[1,:]=[0,1,-l2/2*np.cos(th2),0,0,0,0,0,0]
    Cq[2,:]=[1,0,-l2/2*np.sin(th2),-1,0,-l3/2*np.sin(th3),0,0,0]
    Cq[3,:]=[0,1,l2/2*np.cos(th2),0,-1,l3/2*np.cos(th3),0,0,0]
    Cq[4,:]=[0,0,0,1,0,-l3/2*np.sin(th3),-1,0,0]
    Cq[5,:]=[0,0,0,0,1,l3/2*np.cos(th3),0,-1,0]
    Cq[6,:]=[0,0,0,0,0,0,0,1,0]
    Cq[7,:]=[0,0,0,0,0,0,0,0,1]
    return Cq

def SC_Qe(t):
    global g,Mass,Tau
    m2=Mass[0,0];     m3=Mass[3,3];     m4=Mass[6,6];
    Torque=100*np.sin(Tau*t)
    Qe=np.array([0,-m2*g,Torque,0,-m3*g,0,0,-m4*g,0],dtype=np.float64).T
    return Qe

def SC_Qd(Q_i,dQ_i):
    global l2,l3,N_Constr,N_Coords
    th2 = Q_i[2];   th3 = Q_i[5];
    dth2=dQ_i[2];   dth3=dQ_i[5];
    Qd=np.zeros(N_Constr).T
    Qd[0]=(-dth2**2)*l2/2*np.cos(th2)
    Qd[1]=(-dth2**2)*l2/2*np.sin(th2)
    Qd[2]=(dth2**2)*l2/2*np.cos(th2)+(dth3**2)*l3/2*np.cos(th3)
    Qd[3]=(dth2**2)*l2/2*np.sin(th2)+(dth3**2)*l3/2*np.sin(th3)
    Qd[4]=(dth3**2)*l3/2*np.cos(th3)
    Qd[5]=(dth3**2)*l3/2*np.sin(th3)
    Qd[6]=0
    Qd[7]=0
    return Qd

def dY(t,Y):
    global Mass
    Q_i=Y[0:9];     dQ_i=Y[9:19]
    Cq=SC_Cq(Q_i);  Qe=SC_Qe(t);    Qd=SC_Qd(Q_i,dQ_i);

    # [M,Cq.T;Cq;0]
    A1=np.hstack([Mass,Cq.T])
    A2=np.hstack([Cq,np.zeros((8,8))])
    A=np.vstack([A1,A2])

    # [Qe;Qd]
    b=np.hstack([Qe,Qd])

    #Solve
    x=np.linalg.solve(A,b)

    ddQ_i=x[0:9]
    LagMul=x[9:19]
    Ydot=np.hstack([dQ_i,ddQ_i])
    return Ydot


## Given Conditions and Constraints
N_Constr=8; N_Coords=9;


## Constants
m2=1;   m3=1;   m4=1;
J2=1e-5;    J3=1e-5;    J4=1e-5;
l2=0.15;    l3=0.25;    g=9.81;
H=0.01; #Slider Offset
Tau=np.pi/0.1

Mass=np.diag([m2,m2,J2,m3,m3,J3,m4,m4,J4]);


## Initial DOF
th2_0=np.deg2rad(0)


## Local Coordinates
# u1o=np.array([0,0]).T;  u2o=np.array([-l2/2,0]).T;
# u2a=np.array([l2/2,0]).T;   u3a=np.array([-l3/2,0]).T;
# u3b=np.array([l3/2,0]).T;   u4b=np.array([0,0]).T;


## Kinematics
# Body2
th2=th2_0
Rx2=l2*np.cos(th2)/2;   Ry2=l2*np.sin(th2)/2;
# Body3
th3=np.arcsin((H-l2*np.sin(th2))/l3)
Rx3=l2*np.cos(th2)+l3*np.cos(th3)/2;   Ry3=l2*np.sin(th2)+l3*np.sin(th3)/2;
# Body4
th4=0;
Rx4=l2*np.cos(th2)+l3*np.cos(th3);  Ry4=H;


## Generalized Coordinates
Q_i=np.array([Rx2,Ry2,th2,Rx3,Ry3,th3,Rx4,Ry4,th4]).T
dQ_i=np.zeros(len(Q_i)).T


## Simulation Configs
t=0
endTime=0.5
steps=500;
h=endTime/steps

times=np.array([])

QiLog=np.zeros((len(Q_i),steps+1))
dQiLog=np.zeros((len(Q_i),steps+1))
ddQiLog=np.zeros((len(Q_i),steps+1))


for i in range(steps+1):
    # State Vector
    Y=np.hstack([Q_i,dQ_i])

    # RK4 Time Integrate
    k1=dY(t,Y)
    k2=dY(t+0.5*h,Y+0.5*h*k1)
    k3=dY(t+0.5*h,Y+0.5*h*k2)
    k4=dY(t+h,Y+h*k3)
    Grad=(k1+2*k2+2*k3+k4)/6
    Y_Next=Y+Grad*h

    # Pos, Vel, Acc
    Q_i=Y_Next[0:9]
    dQ_i=Y_Next[9:19]
    ddQ_i=Grad[9:19]

    # Record
    QiLog[:,i]=Q_i
    dQiLog[:,i] = dQ_i
    ddQiLog[:,i]=ddQ_i

    # Update
    times = np.append(times,t)
    Y=Y_Next
    t=t+h
    print(f"Solving t={t:.5f}(sec)")



# Total result DataFrame
QiIdx=['Rx2','Ry2','th2','Rx3','Ry3','th3','Rx4','Ry4','th4']
dQiIdx=['dRx2','dRy2','dth2','dRx3','dRy3','dth3','dRx4','dRy4','dth4']
ddQiIdx=['ddRx2','ddRy2','ddth2','ddRx3','ddRy3','ddth3','ddRx4','ddRy4','ddth4']
Times=pd.DataFrame(times,columns=['Time']).T
QiDF=pd.DataFrame(QiLog,index=QiIdx)
dQiDF=pd.DataFrame(dQiLog,index=dQiIdx)
ddQiDF=pd.DataFrame(ddQiLog,index=ddQiIdx)
Result=pd.concat([Times,QiDF,dQiDF,ddQiDF])
Result=Result.T
print(Result)



# Samples=500
# Idx=['Time']+QiIdx+dQiIdx+ddQiIdx
#
# Sampled_Result=pd.DataFrame(0,index=Idx,columns=range(len(Times)))
# for i in range(Samples):
#     Sampled_Result[i]=Result[i]
#
# print(Sampled_Result)



plt.rcParams['font.family']='Times New Roman'
plt.rcParams['font.size']=14
plt.rcParams['mathtext.fontset']='stix'

fig=plt.figure(figsize=(10,10))

plt.suptitle('Slider Crank Dynamics')

plt.subplot(331)
plt.plot(Result['Time'],Result['Rx3'])
plt.xlabel('Time(sec)')
plt.ylabel(r'$R_x^3(m)$')
plt.grid()

plt.subplot(332)
plt.plot(Result['Time'],Result['Ry3'])
plt.xlabel('Time(sec)')
plt.ylabel(r'$R_y^3(m)$')
plt.grid()

plt.subplot(333)
plt.plot(Result['Time'],Result['th3'])
plt.xlabel('Time(sec)')
plt.ylabel(r'$\theta^3(rad)$')
plt.grid()

plt.subplot(334)
plt.plot(Result['Time'],Result['dRx3'])
plt.xlabel('Time(sec)')
plt.ylabel(r'$\dotR_x^3(m/s)$')
plt.grid()

plt.subplot(335)
plt.plot(Result['Time'],Result['dRy3'])
plt.xlabel('Time(sec)')
plt.ylabel(r'$\dotR_y^3(m/s)$')
plt.grid()

plt.subplot(336)
plt.plot(Result['Time'],Result['dth3'])
plt.xlabel('Time(sec)')
plt.ylabel(r'$\dot\theta^3(rad/s)$')
plt.grid()

plt.subplot(337)
plt.plot(Result['Time'],Result['ddRx3'])
plt.xlabel('Time(sec)')
plt.ylabel(r'$\ddotR_x^3(m/s^2)$')
plt.grid()


plt.subplot(338)
plt.plot(Result['Time'],Result['ddRy3'])
plt.xlabel('Time(sec)')
plt.ylabel(r'$\ddotR_y^3(m/s^2)$')
plt.grid()

plt.subplot(339)
plt.plot(Result['Time'],Result['ddth3'])
plt.xlabel('Time(sec)')
plt.ylabel(r'$\ddot\theta^3(rad/s^2)$')
plt.grid()


plt.tight_layout()
plt.show()