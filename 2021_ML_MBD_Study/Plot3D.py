import numpy as np
import matplotlib.pyplot as plt


K=10

# #균일 추출 시 아래 코드 사용
# Ls=np.linspace(0.1,0.2,K,endpoint=True) # Full 11
# th_0s=np.linspace(-np.pi/2,np.pi/2,K,endpoint=True) # Full 11
# cs=np.linspace(0,0.15,K,endpoint=True) # Full 16
# Ls,th_0s=np.meshgrid(Ls,th_0s)

#랜덤 추출 시 아래 코드 사용
np.random.seed(777)
Ls=np.linspace(0.1,0.2,11,endpoint=True) # Full 11
th_0s=np.linspace(-np.pi/2,np.pi/2,11,endpoint=True) # Full 11
cs=np.linspace(0,0.15,16,endpoint=True) # Full 16
Ls=np.random.choice(Ls,size=K,replace=False)
cs=np.random.choice(cs,size=K,replace=False)
th_0s=np.random.choice(th_0s,size=K,replace=False)
print(Ls)
print(cs)
print(th_0s)

Ls,th_0s=np.meshgrid(Ls,th_0s)


plt.figure(figsize=(8,8))
ax=plt.axes(projection='3d')
Cs=np.zeros((len(Ls),len(th_0s)))
for i in range(len(Ls)):
    for j in range(len(th_0s)):
        for k in range(len(cs)):
            Cs=cs[k]
            ax.scatter(Ls,th_0s,Cs,c='r')


Samples=len(Ls)*len(th_0s)*len(cs)

ax.set_title(f'Rand{K}K Latin Hypercube Sampling, {Samples} Samples, Single Pendulum')
ax.set_xlabel(r'$L(m)$')
ax.set_ylabel(r'$\theta_0(rad)$')
ax.set_zlabel(r'$c$')

ax.set_xlim(0.1,0.2)
ax.set_ylim(-np.pi/2,np.pi/2)
ax.set_zlim(0,0.15)
plt.grid()
# plt.savefig(f'Rand{K}K.png',transparent=False)
# plt.show()


