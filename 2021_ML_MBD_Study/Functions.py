import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import glob
import torch


def RandomUniform(lowbound, upbound, size):
    Array = np.random.random(size)
    Scale = upbound - lowbound
    Shift = lowbound
    Array = np.sort(Array * Scale + Shift)
    return Array


# DataPath('dir')내의 csv파일들을 읽어 단일 판다스 데이터프레임으로 병합하여 반환한다
def CollectConcat(DataPath):
    DataPath += '/*.csv'
    print(f"Building from {DataPath}")
    DataList = glob.glob(DataPath)
    DataFrames = [0] * len(DataList)  # 미리 배열 크기를 할당해둔다
    Keys = [0] * len(DataList)  # 키값
    for i, FileName in enumerate(DataList):
        DataFrames[i] = pd.read_csv(FileName)
        Keys[i] = 'Dataset' + str(i + 1)
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(DataList)} Reading from file: {FileName}")
    Full = pd.concat(DataFrames, axis=0, keys=Keys)
    return Full


# glob.glob으로 읽어들인 GlobDataList에서 RateTrain만큼의 비율을 학습데이터로 선택하고, 학습데이터, 검증데이터 파일 리스트를 반환한다
def SplitData(GlobDataList, RateTrain, RateValid):
    # glob.glob csv 파일들과 분할비율을 인수로 받는다
    DataList = GlobDataList
    NumData = len(DataList)
    NumTrain = int(NumData * RateTrain)
    NumValid = NumData - NumTrain
    TrainIndexes = np.zeros(NumTrain, dtype=int)

    # 비율만큼 학습데이터 무작위로 고르고 인덱스를 따옴
    Train_DataList = np.random.choice(DataList, NumTrain, replace=False)
    for i, data in enumerate(DataList):
        for j, traindata in enumerate(Train_DataList):
            if data == traindata:
                TrainIndexes[j] = i

    # 비율만큼 검증데이터 무작위로 고르고 인덱스를 따옴
    Valid_DataList = np.delete(DataList, TrainIndexes)

    return Train_DataList, Valid_DataList


def MyPlotTemplate():
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 14
    plt.rcParams['mathtext.fontset'] = 'stix'


# DNN 모델 아키텍처
class DNNModel(nn.Module):

    def __init__(self):
        super().__init__()

    def UseTrainData(self, Datapath):
        if type(Datapath) == str:
            self.TrainData = pd.read_csv(Datapath)
        elif type(Datapath) == pd.core.frame.DataFrame:
            self.TrainData = Datapath
        self.__TrainDataMax = self.TrainData.max()
        self.__TrainDataMin = self.TrainData.min()
        self.__TrainDataMean = self.TrainData.mean()
        self.__TrainDataStd = self.TrainData.std()

    def SetInputOutput(self, INcolumns, OUTcolumns):  # 입출력이 Series인 경우 'x', 'y' 식으로 입력
        self.__InputColumns = INcolumns
        self.__OutputColumns = OUTcolumns

        self.__InputTrainData = self.TrainData[self.__InputColumns]
        self.__InputTrainDataMax = self.__InputTrainData.max()
        self.__InputTrainDataMin = self.__InputTrainData.min()
        self.__InputTrainDataMean = self.__InputTrainData.mean()
        self.__InputTrainDataStd = self.__InputTrainData.std()

        self.__OutputTrainData = self.TrainData[self.__OutputColumns]
        self.__OutputTrainDataMax = self.__OutputTrainData.max()
        self.__OutputTrainDataMin = self.__OutputTrainData.min()
        self.__OutputTrainDataMean = self.__OutputTrainData.mean()
        self.__OutputTrainDataStd = self.__OutputTrainData.std()

    def NormalizeInput(self, DF, **kwargs):
        if 'mode' in kwargs and kwargs['mode'] == 'minmax':
            DF = (DF - self.__InputTrainDataMin) / (self.__InputTrainDataMax - self.__InputTrainDataMin)
        if 'mode' in kwargs and kwargs['mode'] == 'gaussian':
            DF = (DF - self.__InputTrainDataMean) / self.__InputTrainDataStd
        return DF

    def NormalizeOutput(self, DF, **kwargs):
        if 'mode' in kwargs and kwargs['mode'] == 'minmax':
            DF = (DF - self.__OutputTrainDataMin) / (self.__OutputTrainDataMax - self.__OutputTrainDataMin)
        if 'mode' in kwargs and kwargs['mode'] == 'gaussian':
            DF = (DF - self.__OutputTrainDataMean) / self.__OutputTrainDataStd
        return DF

    def InverseInput(self, NormalizedDF, **kwargs):
        if 'mode' in kwargs and kwargs['mode'] == 'minmax':
            DF = NormalizedDF * (self.__InputTrainDataMax - self.__InputTrainDataMin) + self.__InputTrainDataMin
        if 'mode' in kwargs and kwargs['mode'] == 'gaussian':
            DF = NormalizedDF * self.__InputTrainDataStd + self.__InputTrainDataMean
        return DF

    def InverseOutput(self, NormalizedDF, **kwargs):
        if 'mode' in kwargs and kwargs['mode'] == 'minmax':
            DF = NormalizedDF * (self.__OutputTrainDataMax - self.__OutputTrainDataMin) + self.__OutputTrainDataMin
        if 'mode' in kwargs and kwargs['mode'] == 'gaussian':
            DF = NormalizedDF * self.__OutputTrainDataStd + self.__OutputTrainDataMean
        return DF

    def Summary(self, **kwargs):
        print(f"\n{'=' * 30:^} MODEL SUMMARY {'=' * 30:^}")
        print(f"TOTAL {self.__Hiddens} HIDDEN LAYERS")
        Modules = list(self.modules())
        for Module in Modules:
            print(Module)
        if 'show_params' in kwargs:
            if kwargs['show_params']:
                print(f"{'=' * 20:^} WEIGHTS AND BIAS {'=' * 20:^}")
                for layername in self.state_dict():
                    print(layername, self.state_dict()[layername])
        print(f"{'=' * 75:^}\n")

    def SetModel(self, **kwargs):
        if 'input_dim' in kwargs.keys():  # 입력 차원
            self.__inD = kwargs['input_dim']
        if 'nodes' in kwargs.keys():  # FC Layer 노드 차원
            self.__Nodes = kwargs['nodes']
        if 'output_dim' in kwargs.keys():  # 출력 차원
            self.__outD = kwargs['output_dim']
        if 'hidden_layers' in kwargs.keys():  # 은닉층 수
            self.__Hiddens = kwargs['hidden_layers']
        if 'activation' in kwargs.keys():  # 활성화함수 타입
            if kwargs['activation'] == 'relu':
                self.__Activation = nn.ReLU()
            if kwargs['activation'] == 'leakyrelu':
                self.__Activation = nn.LeakyReLU()
            if kwargs['activation'] == 'tanh':
                self.__Activation = nn.Tanh()
            if kwargs['activation'] == 'sigmoid':
                self.__Activation = nn.Sigmoid()

        # 레이어 선언
        self.__Input = nn.Linear(self.__inD, self.__Nodes, bias=True)
        self.__HiddenLayers = nn.ModuleList()
        for i in range(self.__Hiddens):
            self.__HiddenLayers.add_module(f"HiddenLayer{i + 1}", nn.Linear(self.__Nodes, self.__Nodes, bias=True))
        self.__Output = nn.Linear(self.__Nodes, self.__outD, bias=True)

    def forward(self, X):  # Feed Forward Network
        X = self.__Input(X)
        for HiddenLayer in self.__HiddenLayers:
            X = HiddenLayer(X)
            X = self.__Activation(X)
        X = self.__Output(X)
        return X


# MBD 적분기
class MBD_Integrator:
    __Example = None
    __g = 9.80665

    def SetGravity(self, GValue):
        self.__g = GValue

    def SC_Kin(self, **kwargs):
        def th(t, tau):
            return t / tau - np.sin(t * tau) / tau ** 2

        def dth(t, tau):
            return 1 / tau - np.cos(t * tau) / tau

        def ddth(t, tau):
            return np.sin(t * tau)

        def pi(t, tau, Lr, r):
            L = Lr * r
            return -np.arcsin((r * np.sin(t / tau - np.sin(t * tau) / tau ** 2)) / L)

        def xb(t, tau, Lr, r):
            L = Lr * r
            return L * (1 - (r ** 2 * np.sin(t / tau - np.sin(t * tau) / tau ** 2) ** 2) / L ** 2) ** (
                    1 / 2) + r * np.cos(t / tau - np.sin(t * tau) / tau ** 2)

        # Symbolic math by MATLAB
        def dpi(t, tau, Lr, r):
            L = Lr * r
            return -(r * np.cos(t / tau - np.sin(t * tau) / tau ** 2) * (1 / tau - np.cos(t * tau) / tau)) / (
                    L * (-(r ** 2 * np.sin((np.sin(t * tau) - t * tau) / tau ** 2) ** 2 - L ** 2) / L ** 2) ** (
                    1 / 2))

        def dxb(t, tau, Lr, r):
            L = Lr * r
            return (r ** 2 * np.sin((np.sin(t * tau) - t * tau) / tau ** 2) * np.cos(
                t / tau - np.sin(t * tau) / tau ** 2) * (1 / tau - np.cos(t * tau) / tau)) / (L * (
                    -(r ** 2 * np.sin((np.sin(t * tau) - t * tau) / tau ** 2) ** 2 - L ** 2) / L ** 2) ** (
                                                                                                      1 / 2)) - r * np.sin(
                t / tau - np.sin(t * tau) / tau ** 2) * (1 / tau - np.cos(t * tau) / tau)

        def ddpi(t, tau, Lr, r):
            L = Lr * r
            return (r * np.sin(t / tau - np.sin(t * tau) / tau ** 2) * (
                    1 / tau - np.cos(t * tau) / tau) ** 2 - r * np.cos(
                t / tau - np.sin(t * tau) / tau ** 2) * np.sin(t * tau) + (
                            dpi(t, tau, Lr, r) * r ** 2 * np.cos(t / tau - np.sin(t * tau) / tau ** 2) * np.sin(
                        t / tau - np.sin(t * tau) / tau ** 2) * (1 / tau - np.cos(t * tau) / tau)) / (L * (
                    -(r ** 2 * np.sin((np.sin(t * tau) - t * tau) / tau ** 2) ** 2 - L ** 2) / L ** 2) ** (
                                                                                                              1 / 2))) / (
                           L * (-(
                           r ** 2 * np.sin((np.sin(t * tau) - t * tau) / tau ** 2) ** 2 - L ** 2) / L ** 2) ** (
                                   1 / 2))

        def ddxb(t, tau, Lr, r):
            L = Lr * r
            return (dpi(t, tau, Lr, r) * r * np.cos(t / tau - np.sin(t * tau) / tau ** 2) * (
                    1 - (r ** 2 * np.sin(t / tau - np.sin(t * tau) / tau ** 2) ** 2) / L ** 2) ** (1 / 2) * (
                            1 / tau - np.cos(t * tau) / tau)) / (
                           -(r ** 2 * np.sin((np.sin(t * tau) - t * tau) / tau ** 2) ** 2 - L ** 2) / L ** 2) ** (
                           1 / 2) - r * np.sin(t / tau - np.sin(t * tau) / tau ** 2) * np.sin(t * tau) - (
                           r * np.sin((np.sin(t * tau) - t * tau) / tau ** 2) * (
                           r * np.sin(t / tau - np.sin(t * tau) / tau ** 2) * (
                           1 / tau - np.cos(t * tau) / tau) ** 2 - r * np.cos(
                       t / tau - np.sin(t * tau) / tau ** 2) * np.sin(t * tau) + (
                                   dpi(t, tau, Lr, r) * r ** 2 * np.cos(
                               t / tau - np.sin(t * tau) / tau ** 2) * np.sin(
                               t / tau - np.sin(t * tau) / tau ** 2) * (
                                           1 / tau - np.cos(t * tau) / tau)) / (L * (-(r ** 2 * np.sin(
                       (np.sin(t * tau) - t * tau) / tau ** 2) ** 2 - L ** 2) / L ** 2) ** (1 / 2)))) / (L * (
                    -(r ** 2 * np.sin((np.sin(t * tau) - t * tau) / tau ** 2) ** 2 - L ** 2) / L ** 2) ** (
                                                                                                                 1 / 2)) - r * np.cos(
                t / tau - np.sin(t * tau) / tau ** 2) * (1 / tau - np.cos(t * tau) / tau) ** 2

        if 'tau' in kwargs.keys():
            self.__tau = kwargs['tau']
        if 'r' in kwargs.keys():
            self.__r = kwargs['r']
        if 'Lr' in kwargs.keys():
            self.__Lr = kwargs['Lr']

        ## INPUT VARS
        # taus = np.linspace(1, 2, 11)  # 11 tau steps
        # rs = np.linspace(1, 3, 11)  # 11 r steps
        # Lrs = np.linspace(2.5, 3.5, 11)  # 11 L/r steps

        ts = np.linspace(0, 5, 501)
        Data = np.zeros((len(ts), 11))
        Columns = ['Time', 'tau', 'r', 'L/r', 'th', 'pi', 'dpi', 'ddpi', 'xb', 'dxb', 'ddxb']
        for i, t in enumerate(ts):
            _th = th(t, self.__tau)
            _pi = pi(t, self.__tau, self.__Lr, self.__r)
            _dpi = dpi(t, self.__tau, self.__Lr, self.__r)
            _ddpi = ddpi(t, self.__tau, self.__Lr, self.__r)
            _xb = xb(t, self.__tau, self.__Lr, self.__r)
            _dxb = dxb(t, self.__tau, self.__Lr, self.__r)
            _ddxb = ddxb(t, self.__tau, self.__Lr, self.__r)
            Data[i, :] = np.array([t, self.__tau, self.__Lr, self.__r, _th, _pi, _dpi, _ddpi, _xb, _dxb, _ddxb])
        Data = pd.DataFrame(Data, columns=Columns)
        return Data

    def Pendulum_Single(self, **kwargs):
        def dY(t, Y):
            th = Y[0];
            dth = Y[1];

            dY = np.zeros(2)
            dY[0] = dth
            dY[1] = -self.__c / (self.__m * self.__L) * dth - self.__g / self.__L * np.sin(th)

            return dY

        ## Input Variables
        # t=[0,2], 200 steps sampled
        if 'L' in kwargs.keys():  # [0.1,0.2] 0.01
            self.__L = kwargs['L']
        if 'c' in kwargs.keys():  # [0,0.15] 0.01
            self.__c = kwargs['c']
        if 'th_0' in kwargs.keys():  # [-pi/2,pi/2], pi/10
            self.__th_0 = kwargs['th_0']

        ## Constants
        self.__m = 0.3;
        self.__dth_0 = np.pi / 2;

        ## Output = th,dth,ddth

        ## State Vector
        Y = np.zeros(2)
        Y[0] = self.__th_0
        Y[1] = self.__dth_0

        ## Simul Configs
        t = 0
        endTime = 2
        steps = 200
        h = endTime / steps
        times = np.array([])

        QiLog = np.zeros(steps + 1)
        dQiLog = np.zeros(steps + 1)
        ddQiLog = np.zeros(steps + 1)

        for i in range(steps + 1):
            # print(f"Solving t={t:.5f}(sec)")
            k1 = dY(t, Y)
            k2 = dY(t + 0.5 * h, Y + k1 * 0.5 * h)
            k3 = dY(t + 0.5 * h, Y + k2 * 0.5 * h)
            k4 = dY(t + h, Y + k3 * h)
            Grad = (k1 + 2 * k2 + 2 * k3 + k4) / 6
            Y_Next = Y + Grad * h

            QiLog[i] = Y_Next[0]
            dQiLog[i] = Y_Next[1]
            ddQiLog[i] = Grad[1]

            Y = Y_Next
            times = np.append(times, t)
            t = t + h

        QiIdx = ['th']
        dQiIdx = ['dth']
        ddQiIdx = ['ddth']
        Times = pd.DataFrame(times, columns=['Time']).T
        QiDF = pd.DataFrame(QiLog, columns=QiIdx).T
        dQiDF = pd.DataFrame(dQiLog, columns=dQiIdx).T
        ddQiDF = pd.DataFrame(ddQiLog, columns=ddQiIdx).T
        Result = pd.concat([Times, QiDF, dQiDF, ddQiDF])
        Result.loc['L'] = self.__L
        Result.loc['c'] = self.__c
        Result.loc['th_0'] = self.__th_0
        Result = Result.T
        return Result

    def Pendulum_Double(self, **kwargs):
        def dY(t, Y):
            th1 = Y[0];
            th2 = Y[1];
            dth1 = Y[2];
            dth2 = Y[3];

            dY = np.zeros(4)
            dY[0] = dth1
            dY[1] = dth2

            Mi = np.zeros((2, 2))
            Mi[0, :] = [(self.__m1 + self.__m2) * self.__l1, self.__m2 * self.__l2 * np.cos(th1 - th2)]
            Mi[1, :] = [self.__m2 * self.__l1 * np.cos(th1 - th2), self.__m2 * self.__l2]

            Qi = np.zeros(2)
            Qi[0] = -(dth2 ** 2) * self.__m2 * self.__l2 * np.sin(th1 - th2) - (
                    self.__m1 + self.__m2) * self.__g * np.sin(th1)
            Qi[1] = (dth1 ** 2) * self.__m2 * self.__l1 * np.sin(th1 - th2) - self.__m2 * self.__g * np.sin(th2)

            dY[2:4] = np.linalg.solve(Mi, Qi)
            return dY

        ## Constants
        self.__m1 = 2;
        self.__m2 = 1;
        self.__th1_0 = 1.6;
        self.__th2_0 = 1.6

        ## Input Variables
        if 'l1' in kwargs.keys():
            self.__l1 = kwargs['l1']
        if 'l2' in kwargs.keys():
            self.__l2 = kwargs['l2']
        if 'dth1_0' in kwargs.keys():
            self.__dth1_0 = kwargs['dth1_0']
        if 'dth2_0' in kwargs.keys():
            self.__dth2_0 = kwargs['dth2_0']
        # t=[0,5], 501 steps sampled
        ## Input Variables
        # l1 = [1,2] 0.1
        # l2 = [2,3] 0.1
        # dth1_0 = [0,0.1] 0.01
        # dth2_0 = [0.3,0.5] 0.02
        ## Output = th1,th2,dth1,dth2

        ## State Vector
        Y = np.zeros(4)
        Y = [self.__th1_0, self.__th2_0, self.__dth1_0, self.__dth2_0]

        ## Simul Configs
        t = 0
        endTime = 5
        steps = 500
        h = endTime / steps
        times = np.array([])

        QiLog = np.zeros((2, steps + 1))
        dQiLog = np.zeros((2, steps + 1))
        ddQiLog = np.zeros((2, steps + 1))

        for i in range(steps + 1):
            # print(f"Solving t={t:.5f}(sec)")
            k1 = dY(t, Y)
            k2 = dY(t + 0.5 * h, Y + k1 * 0.5 * h)
            k3 = dY(t + 0.5 * h, Y + k2 * 0.5 * h)
            k4 = dY(t + h, Y + k3 * h)
            Grad = (k1 + 2 * k2 + 2 * k3 + k4) / 6
            Y_Next = Y + Grad * h

            QiLog[:, i] = Y_Next[0:2]
            dQiLog[:, i] = Y_Next[2:5]
            ddQiLog[:, i] = k1[2:5]

            Y = Y_Next
            times = np.append(times, t)
            t = t + h

        QiIdx = ['th1', 'th2']
        dQiIdx = ['dth1', 'dth2']
        ddQiIdx = ['ddth1', 'ddth2']
        Times = pd.DataFrame(times, columns=['Time']).T
        QiDF = pd.DataFrame(QiLog, index=QiIdx)
        dQiDF = pd.DataFrame(dQiLog, index=dQiIdx)
        ddQiDF = pd.DataFrame(ddQiLog, index=ddQiIdx)
        Result = pd.concat([Times, QiDF, dQiDF, ddQiDF])
        Result.loc['L1'] = self.__l1
        Result.loc['L2'] = self.__l2
        Result.loc['dth1_0'] = self.__dth1_0
        Result.loc['dth2_0'] = self.__dth2_0
        Result = Result.T
        return Result

    def SC_Dyn(self, **kwargs):
        def SC_Cq(Q_i):
            th2 = Q_i[2];
            th3 = Q_i[5]
            Cq = np.zeros((self.__N_Constr, self.__N_Coords))
            Cq[0, :] = [1, 0, self.__l2 / 2 * np.sin(th2), 0, 0, 0, 0, 0, 0]
            Cq[1, :] = [0, 1, -self.__l2 / 2 * np.cos(th2), 0, 0, 0, 0, 0, 0]
            Cq[2, :] = [1, 0, -self.__l2 / 2 * np.sin(th2), -1, 0, -self.__l3 / 2 * np.sin(th3), 0, 0, 0]
            Cq[3, :] = [0, 1, self.__l2 / 2 * np.cos(th2), 0, -1, self.__l3 / 2 * np.cos(th3), 0, 0, 0]
            Cq[4, :] = [0, 0, 0, 1, 0, -self.__l3 / 2 * np.sin(th3), -1, 0, 0]
            Cq[5, :] = [0, 0, 0, 0, 1, self.__l3 / 2 * np.cos(th3), 0, -1, 0]
            Cq[6, :] = [0, 0, 0, 0, 0, 0, 0, 1, 0]
            Cq[7, :] = [0, 0, 0, 0, 0, 0, 0, 0, 1]
            return Cq

        def SC_Qe(t):
            m2 = self.__Mass[0, 0];
            m3 = self.__Mass[3, 3];
            m4 = self.__Mass[6, 6];
            Torque = 100 * np.sin(self.__Tau * t)
            Qe = np.array([0, -m2 * self.__g, Torque, 0, -m3 * self.__g, 0, 0, -m4 * self.__g, 0], dtype=np.float64).T
            return Qe

        def SC_Qd(Q_i, dQ_i):
            th2 = Q_i[2];
            th3 = Q_i[5];
            dth2 = dQ_i[2];
            dth3 = dQ_i[5];
            Qd = np.zeros(self.__N_Constr).T
            Qd[0] = (-dth2 ** 2) * self.__l2 / 2 * np.cos(th2)
            Qd[1] = (-dth2 ** 2) * self.__l2 / 2 * np.sin(th2)
            Qd[2] = (dth2 ** 2) * self.__l2 / 2 * np.cos(th2) + (dth3 ** 2) * self.__l3 / 2 * np.cos(th3)
            Qd[3] = (dth2 ** 2) * self.__l2 / 2 * np.sin(th2) + (dth3 ** 2) * self.__l3 / 2 * np.sin(th3)
            Qd[4] = (dth3 ** 2) * self.__l3 / 2 * np.cos(th3)
            Qd[5] = (dth3 ** 2) * self.__l3 / 2 * np.sin(th3)
            Qd[6] = 0
            Qd[7] = 0
            return Qd

        def dY(t, Y):
            Q_i = Y[0:9];
            dQ_i = Y[9:19]
            Cq = SC_Cq(Q_i);
            Qe = SC_Qe(t);
            Qd = SC_Qd(Q_i, dQ_i);

            # [M,Cq.T;Cq;0]
            A1 = np.hstack([self.__Mass, Cq.T])
            A2 = np.hstack([Cq, np.zeros((8, 8))])
            A = np.vstack([A1, A2])

            # [Qe;Qd]
            b = np.hstack([Qe, Qd])

            # Solve
            x = np.linalg.solve(A, b)

            ddQ_i = x[0:9]
            LagMul = x[9:19]
            Ydot = np.hstack([dQ_i, ddQ_i])
            return Ydot

        ## Given Conditions and Constraints
        self.__N_Constr = 8;
        self.__N_Coords = 9;

        ## Constants
        m2 = 1;
        m3 = 1;
        m4 = 1;
        J2 = 1e-5;
        J3 = 1e-5;
        J4 = 1e-5;
        self.__l2 = 0.15;
        self.__l3 = 0.25;
        H = 0.01;  # Slider Offset
        self.__Tau = np.pi / 0.1

        self.__Mass = np.diag([m2, m2, J2, m3, m3, J3, m4, m4, J4]);

        ## Initial DOF
        th2_0 = np.deg2rad(0)

        ## Local Coordinates
        # u1o=np.array([0,0]).T;  u2o=np.array([-self.__l2/2,0]).T;
        # u2a=np.array([self.__l2/2,0]).T;   u3a=np.array([-self.__l3/2,0]).T;
        # u3b=np.array([self.__l3/2,0]).T;   u4b=np.array([0,0]).T;

        ## Kinematics
        # Body2
        th2 = th2_0
        Rx2 = self.__l2 * np.cos(th2) / 2;
        Ry2 = self.__l2 * np.sin(th2) / 2;
        # Body3
        th3 = np.arcsin((H - self.__l2 * np.sin(th2)) / self.__l3)
        Rx3 = self.__l2 * np.cos(th2) + self.__l3 * np.cos(th3) / 2;
        Ry3 = self.__l2 * np.sin(th2) + self.__l3 * np.sin(th3) / 2;
        # Body4
        th4 = 0;
        Rx4 = self.__l2 * np.cos(th2) + self.__l3 * np.cos(th3);
        Ry4 = H;

        ## Generalized Coordinates
        Q_i = np.array([Rx2, Ry2, th2, Rx3, Ry3, th3, Rx4, Ry4, th4]).T
        dQ_i = np.zeros(len(Q_i)).T

        ## Simulation Configs
        t = 0
        endTime = 0.5
        steps = 500;
        h = endTime / steps

        times = np.array([])

        QiLog = np.zeros((len(Q_i), steps + 1))
        dQiLog = np.zeros((len(Q_i), steps + 1))
        ddQiLog = np.zeros((len(Q_i), steps + 1))

        for i in range(steps + 1):
            # State Vector
            Y = np.hstack([Q_i, dQ_i])

            # RK4 Time Integrate
            k1 = dY(t, Y)
            k2 = dY(t + 0.5 * h, Y + 0.5 * h * k1)
            k3 = dY(t + 0.5 * h, Y + 0.5 * h * k2)
            k4 = dY(t + h, Y + h * k3)
            Grad = (k1 + 2 * k2 + 2 * k3 + k4) / 6
            Y_Next = Y + Grad * h

            # Pos, Vel, Acc
            Q_i = Y_Next[0:9]
            dQ_i = Y_Next[9:19]
            ddQ_i = Grad[9:19]

            # Record
            QiLog[:, i] = Q_i
            dQiLog[:, i] = dQ_i
            ddQiLog[:, i] = ddQ_i

            # Update
            times = np.append(times, t)
            Y = Y_Next
            t = t + h
            # print(f"Solving t={t:.5f}(sec)")

        # Total result DataFrame
        QiIdx = ['Rx2', 'Ry2', 'th2', 'Rx3', 'Ry3', 'th3', 'Rx4', 'Ry4', 'th4']
        dQiIdx = ['dRx2', 'dRy2', 'dth2', 'dRx3', 'dRy3', 'dth3', 'dRx4', 'dRy4', 'dth4']
        ddQiIdx = ['ddRx2', 'ddRy2', 'ddth2', 'ddRx3', 'ddRy3', 'ddth3', 'ddRx4', 'ddRy4', 'ddth4']
        Times = pd.DataFrame(times, columns=['Time']).T
        QiDF = pd.DataFrame(QiLog, index=QiIdx)
        dQiDF = pd.DataFrame(dQiLog, index=dQiIdx)
        ddQiDF = pd.DataFrame(ddQiLog, index=ddQiIdx)
        Result = pd.concat([Times, QiDF, dQiDF, ddQiDF])
        Result = Result.T
        return Result

    def FBar_Kin(self):

        def FB_Drive(t):
            th2 = self.__dth2 * t
            return th2

        def FB_C(Q, t):
            Rx1 = Q[0];
            Ry1 = Q[1];
            th1 = Q[2]
            Rx2 = Q[3];
            Ry2 = Q[4];
            th2 = Q[5]
            Rx3 = Q[6];
            Ry3 = Q[7];
            th3 = Q[8]
            Rx4 = Q[9];
            Ry4 = Q[10];
            th4 = Q[11]

            C = np.zeros(12)
            C[0:3] = [Rx1, Ry1, th1]
            C[3:5] = (np.array([Rx1, Ry1]) + A(th1) @ self.__u1o) - (
                    np.array([Rx2, Ry2]) + A(th2) @ self.__u2o)  # Rev O
            C[5:7] = (np.array([Rx2, Ry2]) + A(th2) @ self.__u2a) - (
                    np.array([Rx3, Ry3]) + A(th3) @ self.__u3a)  # Rev A
            C[7:9] = (np.array([Rx3, Ry3]) + A(th3) @ self.__u3b) - (
                    np.array([Rx4, Ry4]) + A(th4) @ self.__u4b)  # Rev B
            C[9:11] = (np.array([Rx1, Ry1]) + A(th1) @ self.__u1c) - (
                    np.array([Rx4, Ry4]) + A(th4) @ self.__u4c)  # Rev C
            C[11] = th2 - self.__th2_0 - FB_Drive(t)  # Driving Constraint
            return C

        def FB_Cq(Q):
            Rx1 = Q[0];
            Ry1 = Q[1];
            th1 = Q[2];
            Rx2 = Q[3];
            Ry2 = Q[4];
            th2 = Q[5];
            Rx3 = Q[6];
            Ry3 = Q[7];
            th3 = Q[8];
            Rx4 = Q[9];
            Ry4 = Q[10];
            th4 = Q[11];

            Cq = np.zeros((12, 12))
            for i in range(3):
                Cq[i, i] = 1  # Ground
            Cq[3:5, 0:6] = np.hstack([np.eye(2), (At(th1) @ self.__u1o).reshape(2, 1), -np.eye(2),
                                      (-At(th2) @ self.__u2o).reshape(2, 1)])  # R1-R2,RevO
            Cq[5:7, 3:9] = np.hstack([np.eye(2), (At(th2) @ self.__u2a).reshape(2, 1), -np.eye(2),
                                      (-At(th3) @ self.__u3a).reshape(2, 1)])  # R2-R3,RevA
            Cq[7:9, 6:12] = np.hstack([np.eye(2), (At(th3) @ self.__u3b).reshape(2, 1), -np.eye(2),
                                       (-At(th4) @ self.__u4b).reshape(2, 1)])  # R3-R4,RevB
            Cq[9:11, 0:3] = np.hstack([np.eye(2), (At(th1) @ self.__u1c).reshape(2, 1)])
            Cq[9:11, 9:12] = np.hstack([-np.eye(2), (-At(th4) @ self.__u4c).reshape(2, 1)])  # R1-R4,RevC
            Cq[11, 5] = 1  # Driving Constraint
            return Cq

        def FB_Qd(Q, dQ):
            Rx1 = Q[0];
            Ry1 = Q[1];
            th1 = Q[2];
            Rx2 = Q[3];
            Ry2 = Q[4];
            th2 = Q[5];
            Rx3 = Q[6];
            Ry3 = Q[7];
            th3 = Q[8];
            Rx4 = Q[9];
            Ry4 = Q[10];
            th4 = Q[11];
            dth1 = dQ[2];
            self.__dth2 = dQ[5];
            dth3 = dQ[8];
            dth4 = dQ[11];

            Qd = np.zeros((12, 1))
            Qd[5] = self.__dth2 ** 2 * self.__l2 * np.cos(th2)
            Qd[6] = self.__dth2 ** 2 * self.__l2 * np.sin(th2)
            Qd[7] = dth3 ** 2 * self.__l3 * np.cos(th3)
            Qd[8] = dth3 ** 2 * self.__l3 * np.sin(th3)
            Qd[9] = self.__l1 * np.cos(th1) * dth1 ** 2 - self.__l4 * np.cos(th4) * dth4 ** 2
            Qd[10] = self.__l1 * np.sin(th1) * dth1 ** 2 - self.__l4 * np.sin(th4) * dth4 ** 2
            return Qd

        def FB_Ct():
            Ct = np.zeros((12, 1))
            Ct[-1] = -self.__dth2
            return Ct

        def FB_Cqt():
            Cqt = np.zeros((12, 12))
            return Cqt

        def FB_Ctt():
            Ctt = np.zeros((12, 1))
            return Ctt

        # Loop Closure Eqn for th3 th4 solve
        def LC(th3, th4):
            Eqn1 = self.__l2 * np.cos(self.__th2_0) + self.__l3 * np.cos(th3) + self.__l4 * np.cos(
                th4) - self.__l1  # ==0
            Eqn2 = self.__l2 * np.sin(self.__th2_0) + self.__l3 * np.sin(th3) + self.__l4 * np.sin(th4)  # ==0
            Eqn = np.array([Eqn1, Eqn2])
            return Eqn

        # Loop Closure Jacobian
        def LCJ(th3, th4):
            J = np.zeros((2, 2))
            J[0, :] = [-self.__l3 * np.sin(th3), -self.__l4 * np.sin(th4)]
            J[1, :] = [self.__l3 * np.cos(th3), self.__l4 * np.cos(th4)]
            return J

        # RotZ 행렬
        def A(th):
            A = np.zeros((2, 2))
            A[0, 0] = np.cos(th);
            A[0, 1] = -np.sin(th)
            A[1, 0] = np.sin(th);
            A[1, 1] = np.cos(th)
            return A

        # RotZ 행렬 시간미분
        def At(th):
            At = np.zeros((2, 2))
            At[0, :] = [-np.sin(th), -np.cos(th)]
            At[1, :] = [np.cos(th), -np.sin(th)]
            return At

        ## Constants
        self.__l1 = 0.35
        self.__l2 = 0.2
        self.__l3 = 0.35
        self.__l4 = 0.25

        ## Initial Conditions
        self.__dth2 = 5
        self.__th2_0 = np.deg2rad(57.27)

        ## Solve initial th3 th4
        # th3 th4 초기 추측값
        th3_0 = np.deg2rad(0);
        th4_0 = np.deg2rad(200);

        # NR Solve
        for i in range(100):
            Solution = np.array([th3_0, th4_0]) - np.linalg.inv(LCJ(th3_0, th4_0)) @ LC(th3_0, th4_0)
            th3_0 = Solution[0]
            th4_0 = Solution[1]
            if np.linalg.norm(Solution) <= 1e-6:
                break

        print(np.rad2deg(Solution))
        Go = input(f'Initial th3(deg) and th4(deg) are solved for th2={np.rad2deg(self.__th2_0)}(deg). Proceed? : ')
        if Go == 'y' or Go == 'Y':
            # print('Simulation starts.')
            pass
        else:
            quit()

        ## Local Coordinates
        self.__u1o = np.array([0, 0]);
        self.__u2o = np.array([0, 0]);  # O
        self.__u2a = np.array([self.__l2, 0]);
        self.__u3a = np.array([0, 0]);  # A
        self.__u3b = np.array([self.__l3, 0]);
        self.__u4b = np.array([0, 0]);  # B
        self.__u4c = np.array([self.__l4, 0]);
        self.__u1c = np.array([self.__l1, 0]);  # C

        ## Initial Position
        Rx1 = 0;
        Ry1 = 0;
        th1 = 0;
        Rx2 = 0;
        Ry2 = 0;
        th2 = self.__th2_0;
        Rx3 = self.__l2 * np.cos(th2);
        Ry3 = self.__l2 * np.sin(th2);
        th3 = th3_0;
        Rx4 = Rx3 + self.__l3 * np.cos(th3);
        Ry4 = Ry3 + self.__l3 * np.sin(th3);
        th4 = th4_0;
        Q = np.array([Rx1, Ry1, th1, Rx2, Ry2, th2, Rx3, Ry3, th3, Rx4, Ry4, th4]).T

        ## Simulation Configs
        t = 0
        endTime = 1
        steps = 1000;
        h = endTime / steps
        etol = 1e-5  # For NR Solve
        times = np.array([])

        ## Components
        C = FB_C(Q, t)
        Cq = FB_Cq(Q)
        Ct = FB_Ct()
        Cqt = FB_Cqt()
        Ctt = FB_Ctt()

        ## For Record
        QiLog = np.zeros((len(Q), steps + 1))
        dQiLog = np.zeros((len(Q), steps + 1))
        ddQiLog = np.zeros((len(Q), steps + 1))

        for i in range(steps + 1):
            # print(f"Solving t={t:.5f}(sec)")
            ## Newton Raphson Position Solve
            while True:
                Qvar = np.linalg.solve(Cq, -C)
                Q = Q + Qvar
                C = FB_C(Q, t)
                Cq = FB_Cq(Q)
                if np.linalg.norm(Qvar) <= etol or np.linalg.norm(C) <= etol:
                    break
            Record_Q = Q.T

            ## Velocity
            Cq = FB_Cq(Q)
            Ct = FB_Ct()
            dQ = np.linalg.solve(Cq, -Ct)
            Record_dQ = dQ.T

            ## Acceleration
            Qd = FB_Qd(Q, dQ)
            ddQ = np.linalg.solve(Cq, Qd)
            Record_ddQ = ddQ.T

            ## Record
            QiLog[:, i] = Record_Q
            dQiLog[:, i] = Record_dQ
            ddQiLog[:, i] = Record_ddQ

            times = np.append(times, t)
            t = t + h

        QiIdx = ['Rx1', 'Ry1', 'th1', 'Rx2', 'Ry2', 'th2', 'Rx3', 'Ry3', 'th3', 'Rx4', 'Ry4', 'th4']
        dQiIdx = ['dRx1', 'dRy1', 'dth1', 'dRx2', 'dRy2', 'self.__dth2', 'dRx3', 'dRy3', 'dth3', 'dRx4', 'dRy4', 'dth4']
        ddQiIdx = ['ddRx1', 'ddRy1', 'ddth1', 'ddRx2', 'ddRy2', 'dself.__dth2', 'ddRx3', 'ddRy3', 'ddth3', 'ddRx4',
                   'ddRy4',
                   'ddth4']
        Times = pd.DataFrame(times, columns=['Time']).T

        QiDF = pd.DataFrame(QiLog, index=QiIdx)
        dQiDF = pd.DataFrame(dQiLog, index=dQiIdx)
        ddQiDF = pd.DataFrame(ddQiLog, index=ddQiIdx)

        Result = pd.concat([Times, QiDF, dQiDF, ddQiDF])
        Result = Result.T
        return Result


# AutoEncoder 모델 아키텍처
class AEModel(nn.Module):

    def __init__(self):
        super().__init__()

    def UseTrainData(self, Datapath):
        if type(Datapath) == str:
            self.TrainData = pd.read_csv(Datapath)
        elif type(Datapath) == pd.core.frame.DataFrame:
            self.TrainData = Datapath
        self.TrainDataMax = self.TrainData.max()
        self.TrainDataMin = self.TrainData.min()
        self.TrainDataMean = self.TrainData.mean()
        self.TrainDataStd = self.TrainData.std()

    def SetInputOutput(self, INcolumns, OUTcolumns):  # 입출력이 Series인 경우 'x', 'y' 식으로 입력
        self.InputColumns = INcolumns
        self.OutputColumns = OUTcolumns

        self.InputTrainData = self.TrainData[self.InputColumns]
        self.InputTrainDataMax = self.InputTrainData.max()
        self.InputTrainDataMin = self.InputTrainData.min()
        self.InputTrainDataMean = self.InputTrainData.mean()
        self.InputTrainDataStd = self.InputTrainData.std()

        self.OutputTrainData = self.TrainData[self.OutputColumns]
        self.OutputTrainDataMax = self.OutputTrainData.max()
        self.OutputTrainDataMin = self.OutputTrainData.min()
        self.OutputTrainDataMean = self.OutputTrainData.mean()
        self.OutputTrainDataStd = self.OutputTrainData.std()

    def SetNormalization(self, **kwargs):
        if 'mode' in kwargs:
            self.NormMode = kwargs['mode']

    def NormalizeInput(self, DF):
        if self.NormMode == 'minmax':
            DF = (DF - self.InputTrainDataMin) / (self.InputTrainDataMax - self.InputTrainDataMin)
        if self.NormMode == 'gaussian':
            DF = (DF - self.InputTrainDataMean) / self.InputTrainDataStd
        return DF

    def NormalizeOutput(self, DF):
        if self.NormMode == 'minmax':
            DF = (DF - self.OutputTrainDataMin) / (self.OutputTrainDataMax - self.OutputTrainDataMin)
        if self.NormMode == 'gaussian':
            DF = (DF - self.OutputTrainDataMean) / self.OutputTrainDataStd
        return DF

    def InverseInput(self, NormalizedDF):
        if self.NormMode == 'minmax':
            DF = NormalizedDF * (self.InputTrainDataMax - self.InputTrainDataMin) + self.InputTrainDataMin
        if self.NormMode == 'gaussian':
            DF = NormalizedDF * self.InputTrainDataStd + self.InputTrainDataMean
        return DF

    def InverseOutput(self, NormalizedDF):
        if self.NormMode == 'minmax':
            DF = NormalizedDF * (self.OutputTrainDataMax - self.OutputTrainDataMin) + self.OutputTrainDataMin
        if self.NormMode == 'gaussian':
            DF = NormalizedDF * self.OutputTrainDataStd + self.OutputTrainDataMean
        return DF

    def Summary(self, **kwargs):
        print(f"\n{'=' * 30:^} MODEL SUMMARY {'=' * 30:^}")
        print(f"TOTAL {len(self.Layers)} HIDDEN LAYERS")
        Modules = list(self.modules())
        for Module in Modules:
            print(Module)
        if 'show_params' in kwargs:
            if kwargs['show_params']:
                print(f"{'=' * 20:^} WEIGHTS AND BIAS {'=' * 20:^}")
                for layername in self.state_dict():
                    print(layername, self.state_dict()[layername])
        print(f"{'=' * 75:^}\n")

    def SetModel(self, NodeList, **kwargs):
        # self.SetModel([4,3,2,1,2,3,4],activation='relu')
        if 'activation' in kwargs.keys():  # 활성화함수 타입
            if kwargs['activation'] == 'relu':
                self.Activation = nn.ReLU()
            if kwargs['activation'] == 'leakyrelu':
                self.Activation = nn.LeakyReLU()
            if kwargs['activation'] == 'tanh':
                self.Activation = nn.Tanh()
            if kwargs['activation'] == 'sigmoid':
                self.Activation = nn.Sigmoid()
            if kwargs['activation'] == 'hardsigmoid':
                self.Activation = nn.Hardsigmoid()
            if kwargs['activation'] == 'hardtanh':
                self.Activation = nn.Hardtanh()
            if kwargs['activation'] == 'softplus':
                self.Activation = nn.Softplus()
            if kwargs['activation'] == 'softsign':
                self.Activation = nn.Softsign()
            if kwargs['activation'] == 'tanhshrink':
                self.Activation = nn.Tanhshrink()

        # 레이어 선언
        self.UseBias=True
        self.UseNonlinearActivation=True
        self.Layers = nn.ModuleList()
        for layeridx in range(len(NodeList) - 1):
            self.Layers.add_module(f"Layer{layeridx + 1:02d}",
                                   nn.Linear(NodeList[layeridx], NodeList[layeridx + 1], bias=self.UseBias))

    def forward(self, X):  # Feed Forward Network
        for LayerIdx, Layer in enumerate(self.Layers):
            if LayerIdx < len(self.Layers):
                X = Layer(X)
                if self.UseNonlinearActivation:
                    X = self.Activation(X)
            else:
                X = Layer(X)
        return X

    def forward_Encoder(self, X):
        for LayerIdx, Layer in enumerate(self.Layers):
            if (LayerIdx + 1) > int(np.floor(len(self.Layers) / 2)):
                break
            else:
                X = Layer(X)
                X = self.Activation(X)
        return X

    # def forward_Encode(self,X):


if __name__ == '__main__':
    print('Test Running\n')

    MBD = MBD_Integrator()
    # print(MBD.SC_Kin(tau=2.13,r=1.3,Lr=3))
    # print(MBD.Pendulum_Single(L=0.1,c=0.134,th_0=np.pi/2))
    Data=MBD.Pendulum_Double(l1=1, l2=1, dth1_0=0.5, dth2_0=0.5)


    fig1=plt.figure(figsize=(16,10))
    f1=fig1.add_subplot(221)
    f2=fig1.add_subplot(222)
    f3=fig1.add_subplot(223)
    f1.plot(Data['Time'],np.rad2deg(Data['th1']))
    f2.plot(Data['Time'],Data['dth1'])
    f3.plot(Data['Time'],Data['ddth1'])
    f1.plot(Data['Time'], np.rad2deg(Data['th2']))
    f2.plot(Data['Time'], Data['dth2'])
    f3.plot(Data['Time'], Data['ddth2'])
    f1.grid();f2.grid();f3.grid()
    f1.legend(['Angle1','Angle2']),f2.legend(['Angle1','Angle2']),f3.legend(['Angle1','Angle2'])
    fig1.tight_layout()
    
    plt.show()
    # print(MBD.SC_Dyn())
    # print(MBD.FBar_Kin())