# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 11:05:51 2018

@author: Elena
"""
import numpy as np
import File01_simulation_parameters as parameter
para=parameter.parameter
import os
t=os.environ['SLURM_ARRAY_TASK_ID']

rate1=para.get("rate1")
rate2=para.get("rate2")
timesteps=para.get("testtimesteps")
N=para.get("N")
dt=para.get("dt")
taul=para.get("taul")
taus=para.get("taus")
slope=para.get("slope")
threshold=para.get("threshold")
maxfire=para.get("maxfire")
index=para.get("index")
###############################################################################
#generate afferent spike trains 
###############################################################################
#fixed input spiketrain
np.random.seed(100)


x=np.zeros((N,timesteps+500))
x[0:N/2,:]=np.random.binomial(1,rate1*dt,(N/2,timesteps+500))
x[N/2:N,:]=np.random.binomial(1,rate2*dt,(N-N/2,timesteps+500))
    
xeps_pos,xeps_neg,xeps=np.zeros((3,N,timesteps+500))
for time in np.arange(timesteps+500):
        xeps_pos[:,time]=xeps_pos[:,time-1]*np.exp(-dt/taul)+x[:,time]
        xeps_neg[:,time]=xeps_neg[:,time-1]*np.exp(-dt/taus)+x[:,time]
xeps=1./(taul-taus)*(xeps_pos-xeps_neg)
xeps=xeps[:,500:timesteps+500]
    
###############################################################################
#transfer function
###############################################################################
    
def phi(u):
    if u>=0.:
        return(1./(1.+np.exp(-slope*(u-threshold))))# careful! slope appears in f
    else:
        return(np.exp(slope*(u-threshold))/(1.+np.exp(slope*(u-threshold))))
                 
###############################################################################
#output spikes
#######################§$#######################################################       
#random output spiketrain
np.random.seed()

weight_start=np.load("2019-01-08_start_weight.npy")[index]
weight_end=np.load("2019-01-08_end_weight.npy")[index]
wopt=np.load("2019-01-08_target_weight.npy")[index]

weight_start=np.tile(np.reshape(weight_start,(N,1)),(1,timesteps))    
weight_end=np.tile(np.reshape(weight_end,(N,1)),(1,timesteps))  
wopt=np.tile(np.reshape(wopt,(N,1)),(1,timesteps))
V_start=np.einsum("ij,ij->j",weight_start,xeps)
V_end=np.einsum("ij,ij->j",weight_end,xeps)
Vopt=np.einsum("ij,ij->j",wopt,xeps)

output_rate_start=maxfire*np.vectorize(phi)(V_start)
output_rate_end=maxfire*np.vectorize(phi)(V_end)
output_rate_teacher=maxfire*np.vectorize(phi)(Vopt)

output_spikes_start=np.random.binomial(1,output_rate_start*dt,(timesteps,))
output_spikes_end=np.random.binomial(1,output_rate_end*dt,(timesteps,))
output_spikes_teacher=np.random.binomial(1,output_rate_teacher*dt,(timesteps,))

np.save("2019-02-01_membrane_potential_psth_start"+str(t),V_start)
np.save("2019-02-01_output_spikes_psth_start"+str(t),output_spikes_start)

np.save("2019-02-01_membrane_potential_psth_end"+str(t),V_end)
np.save("2019-02-01_output_spikes_psth_end"+str(t),output_spikes_end)

np.save("2019-02-01_membrane_potential_psth_teacher"+str(t),Vopt)
np.save("2019-02-01_output_spikes_psth_teacher"+str(t),output_spikes_teacher)

parameter={"timesteps":timesteps, "dt":dt}
np.save("time_parameters",parameter)
