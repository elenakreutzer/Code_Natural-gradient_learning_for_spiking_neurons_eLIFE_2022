import numpy as np
import File01_simulation_parameters as para
l_intervals=para.parameter.get("l_intervals")
interval=para.parameter.get("interval")
number_instances=1#para.parameter.get("number_instances")
lrn=para.parameter.get("learningrate_natural")
lre=para.parameter.get("learningrate_euclidean")
N=para.parameter.get("N")
naturalgradient,euclideangradient=np.zeros((2,number_instances,N,l_intervals/interval+1))


for index in range(number_instances):
    naturalgradient[index,:]=np.load("2018-08-23_nwpath_rate_error"+str(index+1)+str(lrn)+".npy")
    euclideangradient[index,:]=np.load("2018-08-23_ewpath_rate_error"+str(index+1)+str(lre)+".npy")
naturalgradient_mean=np.mean(naturalgradient,axis=0)
euclideangradient_mean=np.mean(euclideangradient,axis=0)

np.save("2018-08-24_nwpath.npy",naturalgradient_mean)
np.save("2018-08-24_ewpath.npy",euclideangradient_mean)



