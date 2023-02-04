import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 싱글 펜들럼 데이터
NumSamples=np.array([2,3,5,7,9,10])
DataUsage=np.array([0.4,1.4,6.5,17.7,37.7,51.7,100])

Full=np.array([0.9917,0.9860,0.9948]).mean()
FULL=np.zeros(len(DataUsage))
for i in range(len(FULL)):
    FULL[i]=Full

MK_R2_th=np.array([-0.1127,-0.0375,0.1998,0.8763,0.9786,0.9832])
MK_R2_dth=np.array([0.0451,0.0299,0.2402,0.6107,0.9401,0.9784])
MK_R2_ddth=np.array([-0.0203,0.0034,0.1049,0.7988,0.9667,0.9799])
MK=pd.DataFrame({'th':MK_R2_th,'dth':MK_R2_dth,'ddth':MK_R2_ddth})
MK=MK.T


LHS_R2_th=np.array([-0.1245,-0.1332,0.2546,0.9305,0.9912,0.9915])
LHS_R2_dth=np.array([-0.2752,-0.1188,0.2480,0.7269,0.9838,0.9808])
LHS_R2_ddth=np.array([-0.0962,-0.0308,0.1597,0.8826,0.9876,0.9874])
LHS=pd.DataFrame({'th':LHS_R2_th,'dth':LHS_R2_dth,'ddth':LHS_R2_ddth})
LHS=LHS.T

MK[6]=Full
LHS[6]=Full









# # 더블 펜들럼 데이터
# NumSamples=np.array([2,3,5,7])
# DataUsage=np.array([0.1,0.6,4.3,16.4,100])
#
# Full=np.array([0.9970,0.9997,0.9977,0.9977]).mean()
# FULL=np.zeros(len(NumSamples))
# for i in range(len(FULL)):
#     FULL[i]=Full
#
# MK_R2_th1=np.array([-0.0921,0.3641,0.9905,0.9936])
# MK_R2_th2=np.array([-0.0429,0.9294,0.9814,0.9760])
# MK_R2_dth1=np.array([0.0012,0.1974,0.9583,0.9746])
# MK_R2_dth2=np.array([0.1638,0.9065,0.9679,0.9536])
# MK=pd.DataFrame({'th1':MK_R2_th1,'th2':MK_R2_th2,'dth1':MK_R2_dth1,'dth2':MK_R2_dth2})
# MK=MK.T
#
#
# LHS_R2_th1=np.array([-0.0205,0.3559,0.9843,0.9936])
# LHS_R2_th2=np.array([-0.0308,0.8974,0.9759,0.9931])
# LHS_R2_dth1=np.array([0.0143,0.2110,0.9416,0.9835])
# LHS_R2_dth2=np.array([0.1276,0.8506,0.9379,0.9883])
# LHS=pd.DataFrame({'th1':LHS_R2_th1,'th2':LHS_R2_th2,'dth1':LHS_R2_dth1,'dth2':LHS_R2_dth2})
# LHS=LHS.T
#
# # MK[4]=Full
# # LHS[4]=Full





# fig1=plt.figure()
# plt.plot(NumSamples,MK.mean(),'-o')
# plt.plot(NumSamples,LHS.mean(),'-o')
# plt.plot(NumSamples,FULL,c='r')
# plt.legend(['MK Factorial','Latin Hypercube','Full Data'])
# plt.title('Damped Single Pendulum Comparison')
# plt.xlabel('Number of Sampling Points')
# plt.ylabel(r'$R^2 Score (Average)$')
# plt.grid()
# plt.show()



fig2=plt.figure()
plt.plot(DataUsage,MK.mean(),'-o')
plt.plot(DataUsage,LHS.mean(),'-o')
plt.legend(['MK Factorial','Latin Hypercube','Full Data'])
plt.title('Damped Single Pendulum Comparison')
plt.xlabel('Data Usage (%)')
plt.ylabel(r'$R^2 Score (Average)$')
plt.grid()
plt.show()


