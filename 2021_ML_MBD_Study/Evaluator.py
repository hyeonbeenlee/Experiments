import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import Functions as Func
import glob
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score as R2
from sklearn.metrics import mean_squared_error as MSE



def DNN_EvaluateModel(Example,Case,*args):
    np.random.seed(777)
    torch.manual_seed(777)
    MBD = Func.MBD_Integrator()

    if Example == 'Pendulum_Single':
        HiddenLs = 3
        Nodes = 256
        BatchSize = 256
        Epochs = 100
        LearnRate = 1e-3
        DoShuffle = True
        Input = ['Time', 'L', 'c', 'th_0']
        Output = ['th', 'dth', 'ddth']

    elif Example == 'Pendulum_Double':
        HiddenLs = 4
        Nodes = 64
        BatchSize = 1024
        Epochs = 400
        LearnRate = 1e-4
        DoShuffle = True
        Input = ['Time', 'L1', 'L2', 'dth1_0', 'dth2_0']
        Output = ['th1', 'th2', 'dth1', 'dth2']
        AppendixPath = 'MBD Data/Pendulum_Double/Appendix_Pendulum_Double.csv'
        Train_DList = glob.glob(f"MBD Data/Pendulum_Double/Pendulum_Double_{Case}/*.csv")
        ValidDataPath = 'MBD Data/Pendulum_Double/ValidData.csv'

    elif Example == 'SliderCrankKin':
        HiddenLs = 2
        Nodes = 128
        BatchSize = 64
        Epochs = 200
        LearnRate = 1e-3
        Input = ['Time', 'tau', 'r', 'L/r']
        Output = ['th', 'pi', 'dpi', 'ddpi', 'xb', 'dxb', 'ddxb']
        AppendixPath = 'MBD Data/SliderCrankKin/Appendix_SliderCrankKin.csv'
        Train_DList = glob.glob(f"MBD Data/SliderCrankKin/SliderCrank_{Case}/*.csv")
        ValidDataPath = 'MBD Data/SliderCrankKin/ValidData.csv'

    inD, outD = len(Input), len(Output)
    ModelFilePath = f"DNN Model/{Example}/{Case}.pt" #불러오는 모델 파라미터 파일

    # 데이터프레임으로 병합
    TrainData = Func.CollectConcat(f"MBD Data/{Example}/{Case}")
    # ValidData = pd.read_csv(ValidDataPath)
    ValidData = Func.CollectConcat(f"MBD Data/{Example}/TestData")
    # ValidData = MBD.SC_Kin(tau=1.780,r=1.360,Lr=3.050)

    # 랜덤 데이터 추가
    if 'ADDRNDM' in Case:
        TrainData = pd.concat([TrainData, Func.CollectConcat(f'MBD Data/Pendulum_Single/{Case}_AdditionalRandom')],
                              ignore_index=True)

    Model = Func.DNNModel()
    Model.UseTrainData(TrainData)
    Model.SetInputOutput(Input, Output)
    Model.SetModel(input_dim=inD, nodes=Nodes, output_dim=outD, hidden_layers=HiddenLs, activation='relu')
    Optimizer = torch.optim.Adam(Model.parameters(), lr=LearnRate)

    # 입출력 나눔
    TrainX, TrainY = TrainData[Input], TrainData[Output]
    ValidX, ValidY = ValidData[Input], ValidData[Output]
    # 정규화
    TrainX, TrainY = Model.NormalizeInput(TrainX,mode='gaussian'), Model.NormalizeOutput(TrainY,mode='gaussian')
    ValidX, ValidY = Model.NormalizeInput(ValidX,mode='gaussian'), Model.NormalizeOutput(ValidY,mode='gaussian')
    # 넘파이 변환
    TrainX, TrainY = TrainX.to_numpy(), TrainY.to_numpy()
    ValidX, ValidY = ValidX.to_numpy(), ValidY.to_numpy()
    # 텐서 변환, GPU전송
    TrainX, TrainY = torch.FloatTensor(TrainX).cuda(), torch.FloatTensor(TrainY).cuda()
    ValidX, ValidY = torch.FloatTensor(ValidX).cuda(), torch.FloatTensor(ValidY).cuda()
    Model.cuda()

    # 텐서 데이터셋으로 묶음
    TrainSet = TensorDataset(TrainX, TrainY)
    ValidSet = TensorDataset(ValidX, ValidY)

    # 평가모드로 전환 후 평가
    with torch.no_grad():
        Model.eval()
        ModelFile = torch.load(ModelFilePath)
        Model.load_state_dict(ModelFile)

        Labels = torch.zeros(ValidY.shape).cuda()
        Predictions = torch.zeros(ValidY.shape).cuda()

        for idx, ValidData in enumerate(ValidSet):
            validX, validY = ValidData
            Labels[idx, :] = validY
            PredY = Model.forward(validX)
            Predictions[idx, :] = PredY
            if (idx+1)%10000==0:
                print(f"Predicting values {idx + 1}/{len(ValidSet)}")
        print("Prediction finished.")

    ValidX = pd.DataFrame(ValidX.cpu().numpy(), columns=Input)
    Labels = pd.DataFrame(Labels.cpu().numpy(), columns=Output)
    Predictions = pd.DataFrame(Predictions.cpu().numpy(), columns=Output)

    # 역정규화
    ValidX = Model.InverseInput(ValidX,mode='gaussian')
    Labels = Model.InverseOutput(Labels,mode='gaussian')
    Predictions = Model.InverseOutput(Predictions,mode='gaussian')

    ModelScore = pd.DataFrame(np.zeros((2, len(Output))), columns=Output, index=['R2', 'MSE'])
    for col in Output:
        ModelScore[col]['R2'] = R2(Labels[col], Predictions[col])
        ModelScore[col]['MSE'] = MSE(Labels[col], Predictions[col])

    print(f"Valid Data Labels\n{Labels}")
    print(f"Valid Data Predictions\n{Predictions}")
    print(f"Model Scores\n{ModelScore}")

    ## 플로팅!!!
    Func.MyPlotTemplate()

    Title1 = f'{Case} Data Used Model of {Example}\n'
    Title2 = f"Average R2: {ModelScore.loc['R2'].mean():.6f}, Average MSE: {ModelScore.loc['MSE'].mean():.6f}"
    plt.suptitle(Title1 + Title2)

    if Example == 'Pendulum_Single':
        def Plot_Pendulum_Single():
            plt.subplot(221)
            Title1 = r"R2 Value of $\theta$" + f" {ModelScore['th']['R2'].item():.6f}\n"
            Title2 = r"MSE Value of $\theta$" + f" {ModelScore['th']['MSE'].item():.6f}"
            plt.title(Title1 + Title2)
            plt.scatter(Labels['th'], Predictions['th'], s=0.5)
            plt.grid()
    
            plt.subplot(222)
            Title1 = r"R2 Value of $\dot\theta$" + f" {ModelScore['dth']['R2'].item():.6f}\n"
            Title2 = r"MSE Value of $\dot\theta$" + f" {ModelScore['dth']['MSE'].item():.6f}"
            plt.title(Title1 + Title2)
            plt.scatter(Labels['dth'], Predictions['dth'], s=0.5)
            plt.grid()
    
            plt.subplot(223)
            Title1 = r"R2 Value of $\ddot\theta$" + f" {ModelScore['ddth']['R2'].item():.6f}\n"
            Title2 = r"MSE Value of $\ddot\theta$" + f" {ModelScore['ddth']['MSE'].item():.6f}"
            plt.title(Title1 + Title2)
            plt.scatter(Labels['ddth'], Predictions['ddth'], s=0.5)
            plt.grid()
    
            # plt.figure(figsize=(10,7))
            # Title = f'Damped Single Pendulum, {Case} Sample-Learned Model\n'
            # Title += r'$L=0.135(m), c=0.072, \theta_0=\pi/2(rad)$'
            # plt.suptitle(Title)
            #
            # plt.subplot(311)
            # plt.title(r'$R^2=$'+f'{R2_th:.4f}')
            # plt.ylabel(r'$\theta(rad)$')
            # plt.xlabel('Time(sec)')
            # plt.plot(Labels['Time'],Labels['th'],c='b')
            # plt.plot(Predictions['Time'],Predictions['th'],c='r')
            # plt.legend(['Label','Prediction'])
            # plt.grid()
            #
            # plt.subplot(312)
            # plt.title(r'$R^2=$'+f'{R2_dth:.4f}')
            # plt.ylabel(r'$\dot\theta(rad/s)$')
            # plt.xlabel('Time(sec)')
            # plt.plot(Labels['Time'],Labels['dth'],c='b')
            # plt.plot(Predictions['Time'],Predictions['dth'],c='r')
            # plt.legend(['Label','Prediction'])
            # plt.grid()
            #
            # plt.subplot(313)
            # plt.title(r'$R^2=$'+f'{R2_ddth:.4f}')
            # plt.ylabel(r'$\ddot\theta(rad/s^2)$')
            # plt.xlabel('Time(sec)')
            # plt.plot(Labels['Time'],Labels['ddth'],c='b')
            # plt.plot(Predictions['Time'],Predictions['ddth'],c='r')
            # plt.legend(['Label','Prediction'])
            # plt.grid()
        Plot_Pendulum_Single()

    if Example == 'Pendulum_Double':
        def Plot_Pendulum_Double():
            plt.subplot(221)
            plt.title(r"R2 Value of $\theta^1$" + f" {ModelScore['th1']['R2'].item():.6f}")
            plt.scatter(Labels['th1'], Predictions['th1'], s=0.5)
            plt.grid()
    
            plt.subplot(222)
            plt.title(r"R2 Value  of $\theta^2$" + f" {ModelScore['th2']['R2'].item():.6f}")
            plt.scatter(Labels['th2'], Predictions['th2'], s=0.5)
            plt.grid()
    
            plt.subplot(223)
            plt.title(r"R2 Value of $\dot\theta^1$" + f" {ModelScore['dth1']['R2'].item():.6f}")
            plt.scatter(Labels['dth1'], Predictions['dth1'], s=0.5)
            plt.grid()
    
            plt.subplot(224)
            plt.title(r"R2 Value of $\dot\theta^2$" + f" {ModelScore['dth2']['R2'].item():.6f}")
            plt.scatter(Labels['dth2'], Predictions['dth2'], s=0.5)
            plt.grid()
        Plot_Pendulum_Double()

    if Example == 'SliderCrankKin':
        def Plot_SCKin():
            # Output = ['th', 'pi', 'dpi', 'ddpi', 'xb', 'dxb', 'ddxb']
            plt.subplot(331)
            plt.title(r"R2 Value of $\theta$" + f" {ModelScore['th']['R2'].item():.6f}")
            plt.scatter(Labels['th'], Predictions['th'], s=0.5)
            plt.grid()
    
            plt.subplot(332)
            plt.title(r"R2 Value of $\phi$" + f" {ModelScore['pi']['R2'].item():.6f}")
            plt.scatter(Labels['pi'], Predictions['pi'], s=0.5)
            plt.grid()
    
            plt.subplot(335)
            plt.title(r"R2 Value of $\dot\phi$" + f" {ModelScore['dpi']['R2'].item():.6f}")
            plt.scatter(Labels['dpi'], Predictions['dpi'], s=0.5)
            plt.grid()
    
            plt.subplot(338)
            plt.title(r"R2 Value of $\ddot\phi$" + f" {ModelScore['ddpi']['R2'].item():.6f}")
            plt.scatter(Labels['ddpi'], Predictions['ddpi'], s=0.5)
            plt.grid()
    
            plt.subplot(333)
            plt.title(r"R2 Value of $x_b$" + f" {ModelScore['xb']['R2'].item():.6f}")
            plt.scatter(Labels['xb'], Predictions['xb'], s=0.5)
            plt.grid()
    
            plt.subplot(336)
            plt.title(r"R2 Value of $\dotx_b$" + f" {ModelScore['dxb']['R2'].item():.6f}")
            plt.scatter(Labels['dxb'], Predictions['dxb'], s=0.5)
            plt.grid()
    
            plt.subplot(339)
            plt.title(r"R2 Value of $\ddotx_b$" + f" {ModelScore['ddxb']['R2'].item():.6f}")
            plt.scatter(Labels['ddxb'], Predictions['ddxb'], s=0.5)
            plt.grid()
        Plot_SCKin()

    plt.tight_layout()

    if 'show' in args:
        plt.show()
    if 'save' in args:
        ImageDir=f"PlotFigs/{Example}_{Case}.png"
        plt.savefig(ImageDir)
        plt.close()
        print(f"Plot saved. {ImageDir}\n")



DNN_EvaluateModel('Pendulum_Single','2K','show')