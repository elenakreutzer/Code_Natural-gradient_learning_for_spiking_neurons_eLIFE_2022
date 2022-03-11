
#==============================================================================
# Import modules
#==============================================================================
import os                                                    #module needed to run multiple instances on SLURM cluster
import numpy as np                                           #import numpy package
import File01_simulation_parameters as parameter             #import file that contains all simulation parameters

t=os.environ['SLURM_ARRAY_TASK_ID']                          #run multiple instances on a slurm cluster




#==============================================================================
# Set the parameters
#==============================================================================
para=parameter.parameter                                    #see comments in "File01_simulation_parameters" for explanations of the different parameters

#input rates
rate1=para.get("rate1")
rate2=para.get("rate2")

#time constants of the PSP kernel
taum=para.get("taul")
taus=para.get("taus")
intepsquad=para.get("intepsquad")

#timesteps for learning
timesteps=para.get("timesteps")

#Euler timestep
dt=para.get("dt")



#==============================================================================
# Sample Poisson input spike trains from N afferents 
#==============================================================================

#initialize spike trains
x=np.zeros((2,timesteps))
   
#sample Poisson spike train from rates
x[0]=np.random.binomial(1,rate1*dt,(1,timesteps))
x[1]=np.random.binomial(1,rate2*dt,(1,timesteps))

#calculate the tau_m and tau_s part in the bracket of Eqn. 28 seperately 
xepss,xepsm=np.zeros((2,2,timesteps))

for time in np.arange(1,timesteps):
    xepss[:,time]=(np.exp(-dt/taus)*xepss[:,time-1]+x[:,time])
    xepsm[:,time]=(xepsm[:,time-1]*np.exp(-dt/taum)+x[:,time])
	 
    
#calculate USP (see Eqn. 28)
    xeps=(1./(taum-taus))*(xepsm-xepss)	
    

#save the USP trace  for instance "t" (disable if only one instance should be run manually)   
np.save("inputspikes"+str(rate1)+"_"+str(t)+".npy",xeps[0,:]) 
np.save("inputspikes"+str(rate2)+"_"+str(t)+".npy",xeps[1,:]) 

