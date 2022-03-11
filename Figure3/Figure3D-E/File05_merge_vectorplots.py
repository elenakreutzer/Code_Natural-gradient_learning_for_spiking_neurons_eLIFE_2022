import numpy as np
import File01_parameter_vectorplot as parameter
para=parameter.parameter
precision=para["precision"]
N=para["N"]
samplenumber=para["samplenumber"]
wmin=para["wmin"]
wmax=para["wmax"]
precisioncost=para["precisioncost"]

natvec,eucvec=np.zeros((2,samplenumber,len(np.arange(wmin,wmax,precision)),len(np.arange(wmin,wmax,precision)),N))
print(natvec.shape)
for index in range(samplenumber):
	natvec[index,:,:,:]=np.load("2019-01-07_natvec"+str(index+1)+".npy")
	eucvec[index,:,:,:]=np.load("2019-01-07_eucvec"+str(index+1)+".npy")
natvec_mean=np.mean(natvec,axis=0)
eucvec_mean=np.mean(eucvec,axis=0)
print(natvec_mean)
print(eucvec_mean)

np.save("2019-01-07_vectorplot_nat.npy",natvec_mean)
np.save("2019-01-07_vectorplot_euc.npy",eucvec_mean)


cost=np.zeros((samplenumber,np.shape(np.arange(wmin,wmax,precisioncost))[0],np.shape(np.arange(wmin,wmax,precisioncost))[0]))
for index in range(samplenumber):
    cost[index,:,:]=np.load("2019-01-07_cost"+str(index+1)+".npy")
cost_mean=np.mean(cost,axis=0)

np.save("2019-01-07_contour_cost.npy",cost_mean)
