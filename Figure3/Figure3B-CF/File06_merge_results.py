import numpy as np                                                             #import numpy package
import File01_simulation_parameters as para                                            #import file that contains all simulation parameters

l_intervals=para.parameter.get("l_intervals")                                  #see comments in "File01_simulation_parameters" for explanations of the different parameters
interval=para.parameter.get("interval")
number_instances=para.parameter.get("number_instances")
lrn=para.parameter.get("learningrate_natural")
lra=para.parameter.get("learningrate_approx")
lre=para.parameter.get("learningrate_euclidean")
N=para.parameter.get("N")

naturalgradient,approxgradient,euclideangradient,nat_KL,approx_KL,euclidean_KL,e_wdist,n_wdist=np.zeros((8,number_instances,l_intervals/interval+1))
wstart,wend,wtarget=np.zeros((3,number_instances,N))
for index in range(number_instances):
    #import simulation results for firing rate error
    naturalgradient[index,:]=np.load("2018-08-23_ngradient_rate_error"+str(index+1)+str(lrn)+".npy")       
    approxgradient[index,:]=np.load("2018-08-23_approx_rate_error"+str(index+1)+str(lra)+".npy")
    euclideangradient[index,:]=np.load("2018-08-23_egradient_rate_error"+str(index+1)+str(lre)+".npy")

    #import simulation results for KL cost
    nat_KL[index,:]=np.load("2018-08-23_ngradient_KL_error"+str(index+1)+str(lrn)+".npy")
    approx_KL[index,:]=np.load("2018-08-23_approx_KL_error"+str(index+1)+str(lra)+".npy")
    euclidean_KL[index,:]=np.load("2018-08-23_egradient_KL_error"+str(index+1)+str(lre)+".npy")
    #import simulation results for weight distance
    e_wdist[index,:]=np.load("2021-08-21_e_weightdist"+str(index+1)+str(lre)+".npy")
    n_wdist[index,:]=np.load("2021-08-21_n_weightdist"+str(index+1)+str(lrn)+".npy")
    
    #import start and end weights
    wstart[index,:]=np.load("2018-08-23_wstart"+str(index+1)+".npy")
    wend[index,:]=np.load("2018-08-23_nwpath_rate_error"+str(index+1)+str(lrn)+".npy")[:,-1]
    wtarget[index,:]=np.load("2018-08-23_wtarget"+str(index+1)+".npy")

#calculate mean over all trials for natural-gradient descent
naturalgradient_mean=np.mean(naturalgradient,axis=0)
nat_KL_mean=np.mean(nat_KL,axis=0)
n_wdist_mean=np.mean(n_wdist,axis=0)

#calculate mean over all trials for approximated learning rule
approxgradient_mean=np.mean(approxgradient,axis=0)
approx_KL_mean=np.mean(approx_KL,axis=0)

#calculate mean over all trials for Euclidean-gradient descent
euclideangradient_mean=np.mean(euclideangradient,axis=0)
euclidean_KL_mean=np.mean(euclidean_KL,axis=0)
e_wdist_mean=np.mean(e_wdist,axis=0)

#save the mean firing rate error
np.save("2018-08-24_naturalgradient_rate_error.npy",naturalgradient_mean)
np.save("2018-08-24_approxgradient_rate_error.npy",approxgradient_mean)
np.save("2018-08-24_euclideangradient_rate_error.npy",euclideangradient_mean)

#save the mean KL 
np.save("2018-08-24_naturalgradient_KL_error.npy",nat_KL_mean)
np.save("2018-08-24_approxgradient_KL_error.npy",approx_KL_mean)
np.save("2018-08-24_euclideangradient_KL_error.npy",euclidean_KL_mean)

#calculate the standard deviation in firing rate error
naturalgradient_std=np.std(naturalgradient,axis=0)
approxgradient_std=np.std(approxgradient,axis=0)
euclideangradient_std=np.std(euclideangradient,axis=0)

#calculate the minimum KL	
nat_KL_min=np.min(nat_KL,axis=0)
euclidean_KL_min=np.min(euclidean_KL,axis=0)

#save the minimum KL
np.save("2018-08-24_naturalgradient_KL_error_min.npy",nat_KL_min)
np.save("2018-08-24_euclideangradient_KL_error_min.npy",euclidean_KL_min)

#calculate the maximum KL
nat_KL_max=np.max(nat_KL,axis=0)
euclidean_KL_max=np.max(euclidean_KL,axis=0)

#calculate the maximum KL
np.save("2018-08-24_naturalgradient_KL_error_max.npy",nat_KL_max)
np.save("2018-08-24_euclideangradient_KL_error_max.npy",euclidean_KL_max)

#save the maximum KL
np.save("2021-08-21_n_wdist_mean.npy",n_wdist_mean)
np.save("2021-08-21_e_wdist_mean.npy",e_wdist_mean)

#save start, end and target weight for psth

np.save("2019-01-08_start_weight.npy",wstart)
np.save("2019-01-08_end_weight.npy",wend)
np.save("2019-01-08_target_weight",wtarget)
