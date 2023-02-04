import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import glob

Appendix=pd.read_csv('MBD Data/SliderCrankKin/Appendix_SliderCrankKin.csv')
ValidDataList=Appendix['Valid'].dropna()
ValidData=pd.DataFrame()
for i,ValidDataPath in enumerate(ValidDataList):
    Oldpath='MBD Data/SliderCrank_Full\\'
    Newpath='MBD Data/SliderCrankKin/SliderCrank_Full\\'
    ValidDataPath=ValidDataPath.replace(Oldpath,Newpath)
    data=pd.read_csv(ValidDataPath)
    ValidData=pd.concat([ValidData,data],ignore_index=True)
    print(f"{i+1}/{len(ValidDataList)}")

ValidData.to_csv('MBD Data/SliderCrankKin/ValidData.csv',index=False)



