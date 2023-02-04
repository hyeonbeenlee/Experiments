from DNN import DNN_Train_Save

## 모델 변수 선언
Examples = ['SliderCrankKin']
Cases = ['2K','LHC2K','3K','LHC3K','5K','LHC5K','7K','LHC7K','Full']


for Example in Examples:
    for Case in Cases:
        DNN_Train_Save('Pendulum_Single','2K',generate_data=True,end_evaluate=False,save=False)
