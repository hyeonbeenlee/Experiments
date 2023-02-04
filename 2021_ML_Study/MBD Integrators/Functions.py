import numpy as np

# RotZ 행렬
def A(th):
    A=np.zeros((2,2))
    A[0,0]=np.cos(th);  A[0,1]=-np.sin(th)
    A[1,0]=np.sin(th);  A[1,1]=np.cos(th)
    return A

# RotZ 행렬 시간미분
def At(th):
    At=np.zeros((2,2))
    At[0,:]=[-np.sin(th),-np.cos(th)]
    At[1,:]=[np.cos(th),-np.sin(th)]
    return At


