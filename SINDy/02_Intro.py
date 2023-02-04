import pysindy as ps
import sys
import numpy as np
import matplotlib.pyplot as plt
import MyPackage as mp
sys.path.append("./examples/2_introduction_to_sindy")
if __name__ == 'testing':
    from mock_data import gen_data1, gen_data2
else:  # __main__ import
    from example_data import gen_data1, gen_data2

# train stage
t, x, y = gen_data1()
X = np.stack([x, y], axis=1)
differentiation_method = ps.FiniteDifference(order=2)
feature_library = ps.PolynomialLibrary(degree=3)
optimizer = ps.STLSQ(threshold=0.2)
model = ps.SINDy(differentiation_method=differentiation_method,
                 feature_library=feature_library,
                 optimizer=optimizer,
                 feature_names=['x', 'y'])
model.fit(X, t=t)
model.print()

# test stage
x0, y0, t_test, x_test, y_test = gen_data2()
sim = model.simulate([x0, y0], t=t_test)

mp.visualize.PlotTemplate()
fig, axes = plt.subplots(1, 1)
axes.plot(x0, y0, marker='o', linestyle='none', markerfacecolor='none', markeredgecolor='r', markersize=16, label='Initial Condition')
axes.plot(x_test, y_test, linewidth=3, linestyle='solid', color='k', label='Test data')
axes.plot(sim[:, 0], sim[:, 1], linewidth=2, linestyle='dashed', color='r', label='SINDy model integration')
axes.set(xlabel='x',ylabel='y')
leg = axes.legend(loc=1)
mp.visualize.IncreaseLegendLinewidth(leg)
fig.tight_layout()
plt.show()
