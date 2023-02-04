import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import Functions as Func

Labels=pd.read_csv('Labels.csv').drop(columns='Unnamed: 0')
Predictions=pd.read_csv('Predictions.csv').drop(columns='Unnamed: 0')

Time=np.linspace(0,10,len(Labels),endpoint=True)
Freq=np.fft.rfftfreq(len(Time),Time[1]-Time[0])

plt.figure(figsize=(15,8))
Func.MyPlotTemplate()
for i in range(Labels.shape[1]):
	FFTlabel=np.abs(np.fft.rfft(Labels.to_numpy()[:,i])/len(Labels))
	FFTpred=np.abs(np.fft.rfft(Predictions.to_numpy()[:,i]))/len(Predictions)
	
	plt.subplot(3, 2, i + 1)
	plt.plot(Freq, FFTlabel, c='k', label='Label',zorder=1)
	plt.plot(Freq, FFTpred, c='r', label='Prediction',zorder=0)
	plt.title(Labels.columns[i])
	plt.xlim(0,20), plt.ylim(0)
	plt.grid(),plt.legend()
	plt.xlabel('Frequency (Hz)')
plt.tight_layout()

plt.figure(figsize=(15,8))
Func.MyPlotTemplate()
for i in range(Labels.shape[1]):
	plt.subplot(3, 2, i + 1)
	plt.plot(Time,Labels.iloc[:,i], c='k', label='Label')
	plt.plot(Time, Predictions.iloc[:,i], c='r', label='Prediction')
	plt.title(Labels.columns[i])
	plt.grid(), plt.legend(fontsize=10,loc=1)
	plt.xlabel('Time (sec)')
plt.tight_layout()
plt.show()



