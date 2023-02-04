from Evaluator import DNN_EvaluateModel


Examples = ['Pendulum_Single']
Cases=['2K+L','2K+2L','2K+5L','2K+c','2K+2c','2K+5c','2K+th_0','2K+2th_0','2K+5th_0']

for Example in Examples:
    for Case in Cases:
        DNN_EvaluateModel(Example,Case,'save')




Examples=['SliderCrankKin']
Cases = ['2K','LHC2K','3K','LHC3K','5K','LHC5K','7K','LHC7K','Full']
for Example in Examples:
    for Case in Cases:
        DNN_EvaluateModel(Example,Case,'save')