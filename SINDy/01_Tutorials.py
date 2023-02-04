import numpy as np
import pysindy as ps
import MyPackage
import matplotlib.pyplot as plt
from scipy.integrate import odeint

"""
de Silva, B. M., Champion, K., Quade, M., Loiseau, J. C., Kutz, J. N., & Brunton, S. L. (2020).
Pysindy: a python package for the sparse identification of nonlinear dynamics from data. arXiv preprint arXiv:2004.08424.
"""

dt = 0.002
eps = 1e-8

def lorenz_odeint(X: np.array, t):
    x = X[0]
    y = X[1]
    z = X[2]
    xDot = 10 * (y - x)
    yDot = x * (28 - z) - y
    zDot = x * y - 8 / 3 * z
    XDot = np.array([xDot, yDot, zDot])
    return XDot

def lorenz(X: np.array, t):
    X = X.reshape(-1, 3)
    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]
    xDot = 10 * (y - x)
    yDot = x * (28 - z) - y
    zDot = x * y - 8 / 3 * z
    XDot = np.stack([xDot, yDot, zDot], axis=1)
    return XDot

def generate_training_data():
    global dt, eps
    t = np.arange(0, 10 + eps, dt)
    X_0 = np.array([-8, 8, 27])
    X = odeint(lorenz_odeint, X_0, t)
    return X

def visualize_3Ddata(coordinates: list = ['$x$', '$y$', '$z$'], savePrefix: str = None, time: np.array = None, **kwargs_data):
    if type(time) == None:
        raise UserWarning("Time array is not given.")
    MyPackage.visualize.PlotTemplate()
    colors = ['black', 'red']  # RGBA
    linestyles = ['solid', 'dashed']
    linewidths = [1.5, 1]
    # 2D
    fig1, axes1 = plt.subplots(3, 1, figsize=(10, 10))
    for idx_subplot in range(3):
        for idx_data, (key, data) in enumerate(kwargs_data.items()):
            axes1[idx_subplot].plot(time,
                                    data[:, idx_subplot],
                                    linewidth=linewidths[idx_data],
                                    linestyle=linestyles[idx_data],
                                    color=colors[idx_data],
                                    label=key)
        axes1[idx_subplot].set_xlabel('$t$', fontsize=20)
        axes1[idx_subplot].set_ylabel(coordinates[idx_subplot], fontsize=20)
        axes1[idx_subplot].set_xticks(np.linspace(time[0], time[-1], 6, endpoint=True, dtype=int))
        if idx_subplot == 0:
            leg = axes1[idx_subplot].legend(loc=1, fontsize=13)
            MyPackage.visualize.IncreaseLegendLinewidth(leg)
    fig1.tight_layout()
    # 3D
    fig2, axes2 = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={'projection': '3d'})
    for idx_data, (key, data) in enumerate(kwargs_data.items()):
        axes2.plot(data[:, 0],
                   data[:, 1],
                   data[:, 2],
                   linewidth=linewidths[idx_data],
                   linestyle=linestyles[idx_data],
                   color=colors[idx_data],
                   label=key)
    axes2.set_xlabel(coordinates[0], fontsize=20)
    axes2.set_ylabel(coordinates[1], fontsize=20)
    axes2.set_zlabel(coordinates[2], fontsize=20)
    leg = axes2.legend(loc=1, fontsize=20)
    MyPackage.visualize.IncreaseLegendLinewidth(leg)
    fig2.tight_layout()
    if savePrefix:
        MyPackage.visualize.SaveAllActiveFigures(savePrefix)
    else:
        plt.show()
    plt.close('all')

def Sec_4_1():
    global dt, eps
    X_train = generate_training_data()
    visualize_3Ddata(time=np.arange(0, 10 + eps, dt), Data=X_train)
    
    # invoke model
    model = ps.SINDy()
    model.fit(X_train, dt)
    model.print()
    
    # test dataset
    t_test = np.arange(0, 15 + eps, dt)  # time array
    X_0_test = np.array([8, 7, 15])  # initial condition
    # X_test = odeint(lorenz_odeint, X_0_test, t_test)  # odeint integrations
    _, X_test, XDot_test = MyPackage.integrate.rkint(lorenz_rkint, np.zeros(3), X_0_test, t_test, method='rk5')  # my rk5 integrations
    # XDot_test = model.differentiate(X_test, t=dt)  # from numerical diff
    XDot_test = lorenz(X_test, t_test)  # from true equation
    
    # predict
    XDot_test_pred = model.predict(X_test)  # from trained SINDy
    X_test_pred = model.simulate(X_0_test, t_test)  # integrated from SINDy derivative predictions
    
    # visualize
    visualize_3Ddata(savePrefix='LorenzX', time=t_test, Data=X_test, Prediction=X_test_pred)
    visualize_3Ddata(['$\dot{x}$', '$\dot{y}$', '$\dot{z}$'], savePrefix='LorenzXDot', time=t_test, Data=XDot_test, Prediction=XDot_test_pred)

def Sec_4_2():
    global dt, eps
    X_train = generate_training_data()
    diff_method = ps.FiniteDifference(order=1)
    feature_lib = ps.PolynomialLibrary(degree=3, include_bias=False)
    optimizer = ps.SR3(threshold=0.1, nu=1, tol=1e-6)
    model = ps.SINDy(differentiation_method=diff_method, feature_library=feature_lib, optimizer=optimizer, feature_names=['x', 'y', 'z'])
    model.fit(X_train, t=dt)
    model.print()

def lorenz_rkint(t, X, XDot):
    x = X[0]
    y = X[1]
    z = X[2]
    xDot = XDot[0]
    yDot = XDot[1]
    zDot = XDot[2]
    xDDot = 10 * (yDot - xDot)
    yDDot = xDot * (28 - zDot) - yDot
    zDDot = xDot * yDot - 8 / 3 * zDot
    XDDot = np.array([xDDot, yDDot, zDDot])
    return XDDot

def testIntegrate():
    from scipy.interpolate import Akima1DInterpolator as akispl
    t = np.arange(0, 10 + 1e-8, 1e-2)
    X_0 = np.array([0, 0, 0])
    XDot_0 = np.array([8, 7, 15])
    _, X, XDot = MyPackage.integrate.rkint_detrended(lorenz_rkint, X_0, XDot_0, t, method='rk5')
    
    visualize_3Ddata(X=_, time=t)
    visualize_3Ddata(XDot=X, time=t)
    visualize_3Ddata(XDDot=XDot, time=t)



if __name__ == '__main__':
    # Sec_4_1()
    # Sec_4_2()
    testIntegrate()