import numpy as np
import File01_simulation_parameters as parameter
para=parameter.parameter
rate1=para.get('rate1')
rate2=para.get('rate2')

inputspikes=np.zeros((2*para["sample_spiketrains"],para["timesteps"]))
for i in range(para["sample_spiketrains"]):
	inputspikes[i,:]=np.load("inputspikes"+str(rate1)+"_"+str(i+1)+".npy")
	inputspikes[i+para["sample_spiketrains"],:]=np.load("inputspikes"+str(rate2)+"_"+str(i+1)+".npy")
np.save("inputspikes"+str(rate1)+str(rate2)+".npy",inputspikes)
