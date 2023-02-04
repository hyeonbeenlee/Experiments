import Functions as Func
import numpy as np
import pandas as pd
import torch

MBD = Func.MBD_Integrator()
MBD.SetGravity(9.81)


# ######################SC_KIN##########################
# ## INPUT VARS
# # taus = np.linspace(1, 2, 11)  # 11 tau steps
# # rs = np.linspace(1, 3, 11)  # 11 r steps
# # Lrs = np.linspace(2.5, 3.5, 11)  # 11 L/r steps
# np.random.seed(777)
# taus = np.random.rand(5) * 1 + 1  # *Range + Min boundary
# rs = np.random.rand(5) * 2 + 1
# Lrs = np.random.rand(5) * 1 + 2.5
# Dir = 'MBD Data/SliderCrankKin/TestData/'
# Index = 0
# for tau in taus:
#     for r in rs:
#         for Lr in Lrs:
#             Index += 1
#             Data = MBD.SC_Kin(tau=tau, r=r, Lr=Lr)
#             Data.to_csv(Dir + f"tau={tau:.3f},r={r:.3f},Lr={Lr:.3f}.csv", index=False)
#             print(f"Data {Index} saved.")




######################PENDULUM_DOUBLE##########################
## Input Variables
np.random.seed(777)
l1s = np.random.rand(5) * 1 + 1  # *Range + Min boundary
l2s = np.random.rand(5) * 1 + 2
dth1s = np.random.rand(5) * 0.1 + 0
dth2s = np.random.rand(5) * 0.2 + 0.3
Dir = 'MBD Data/Pendulum_Double/TestData/'
Index = 0
for l1 in l1s:
    for l2 in l2s:
        for dth1 in dth1s:
            for dth2 in dth2s:
                Index += 1
                Data = MBD.Pendulum_Double(l1=l1, l2=l2, dth1_0=dth1, dth2_0=dth2)
                Data.to_csv(Dir + f"l1={l1:.3f},l2={l2:.3f},dth1_0={dth1:.3f},dth2_0={dth2:.3f}.csv", index=False)
                print(f"Data {Index} saved.")

# print(MBD.SC_Kin(tau=2.13,r=1.3,Lr=3))
# print(MBD.Pendulum_Single(L=0.1,c=0.134,th_0=np.pi/2))
# print(MBD.Pendulum_Double(l1=1.4, l2=2.3, dth1_0=0.6, dth2_0=0.3))
# print(MBD.SC_Dyn())
# print(MBD.FBar_Kin())
