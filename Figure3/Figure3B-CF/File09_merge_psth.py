import numpy as np
import File01_simulation_parameters as parameter
para=parameter.parameter
N=para["N"]
samplenumber=3000
timesteps=para["testtimesteps"]
dt=para["dt"]



V_start,V_end,V_teacher,output_spikes_start,output_spikes_end,output_spikes_teacher=np.zeros((6,samplenumber,timesteps))
for index in range(samplenumber):
	V_start[index,:]=np.load("2019-02-01_membrane_potential_psth_start"+str(index+1)+".npy")
	output_spikes_start[index,:]=np.load("2019-02-01_output_spikes_psth_start"+str(index+1)+".npy")
	V_end[index,:]=np.load("2019-02-01_membrane_potential_psth_end"+str(index+1)+".npy")
	output_spikes_end[index,:]=np.load("2019-02-01_output_spikes_psth_end"+str(index+1)+".npy")
	V_teacher[index,:]=np.load("2019-02-01_membrane_potential_psth_teacher"+str(index+1)+".npy")
	output_spikes_teacher[index,:]=np.load("2019-02-01_output_spikes_psth_teacher"+str(index+1)+".npy")

np.save("2019-02-02_membrane_potential_psth_start.npy",V_start)
np.save("2019-02-02_output_spikes_psth_start.npy",output_spikes_start)
np.save("2019-02-02_membrane_potential_psth_end.npy",V_end)
np.save("2019-02-02_output_spikes_psth_end.npy",output_spikes_end)
np.save("2019-02-02_membrane_potential_psth_teacher.npy",V_teacher)
np.save("2019-02-02_output_spikes_psth_teacher.npy",output_spikes_teacher)

	



