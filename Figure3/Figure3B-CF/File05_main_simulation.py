# -* coding: utf-8 -*-
"""
Created on Mon May 23 11:09:39 2016

@author: kreutzer
"""
#==============================================================================
# Import modules
#==============================================================================
import os                                                                      #module needed to run multiple instances on SLURM cluster
import numpy as np                                                             #import numpy package
import File01_simulation_parameters as parameter                               #import file that contains all simulation parameters
t=os.environ['SLURM_ARRAY_TASK_ID']                                          #run multiple instances on a slurm cluster, disable if only a single instance should be run manually




#==============================================================================
# Set the parameters
#==============================================================================
para=parameter.parameter                                                       #see comments in "File01_simulation_parameters" for explanations of the different parameters

#learning rates
lra=para.get("learningrate_approx")
lrn=para.get("learningrate_natural")
lre=para.get("learningrate_euclidean")

#number of neurons
N=para.get("N")

#weight range
wmin=para.get("wmin")
wmax=para.get("wmax")

#input rates
rate1=para.get("rate1")
rate2=para.get("rate2")

#number of afferent spike train samples
sample_spiketrains=para.get("sample_spiketrains")

#time constants of the PSP kernel
taul=para.get("taul")
taus=para.get("taus")
intepsquad=para.get("intepsquad")
inteps=para.get("inteps")

#intervals for learning
l_intervals=para.get("l_intervals")

#timesteps for learning
timesteps=para.get("timesteps")

#Euler timestep
dt=para.get("dt")

#scale for nonlinearity
maxfire=para.get("maxfire")
threshold=para.get("threshold")
slope=para.get("slope")

#lookup table file
lookup_table_file=para.get("lt_file")

#target weights for weightpath plot, usually not necessary
target1=para.get("target1")
target2=para.get("target2")

#intitial weights for weightpath plot, usually not necessary
initial1=para.get("initial1")
initial2=para.get("initial2")

#attenuation for dendritic parametrization
alpha1=para.get("alpha1")
alpha2=para.get("alpha2")

#interval for which to calculate errors
interval=para.get("interval")

#timesteps and samplesize for test set 
testtimesteps=para.get("testtimesteps")
testsample=para.get("testsample")




###############################################################################
#sigmoidal nonlinearity
###############################################################################

#sigmoidal
def phi(u):
    if u>=0.:
        return(1./(1.+np.exp(-slope*(u-threshold))))#threshold=center sigmoidal -resting_potential
    else:
        return(np.exp(slope*(u-threshold))/(1.+np.exp(slope*(u-threshold))))


#vectorize nonlinearity
vphi=np.vectorize(phi)


#derivative of nonlinearity
def phip(u):
      return(slope*phi(u)*(1.-phi(u)))

#==============================================================================
# Define target weight and initial weight
#==============================================================================

#sample both initial and target weight from a uniform distribution
wopt=1./N*np.random.uniform(wmin,wmax,N)
wstart=1./N*np.random.uniform(wmin,wmax,N)

#save both initial and target weight
np.save("2018-08-23_wstart"+str(t)+".npy",wstart)
np.save("2018-08-23_wtarget"+str(t)+".npy",wopt)

#options for fixed initial weight and target to get example weightpath
#wopt=1./N*np.concatenate((target1*np.ones(N/2),target2*np.ones(N-N/2)))
#wstart=1./N *np.concatenate((initial1*np.ones(N/2),initial2*np.ones(N-N/2)))

#attenuation_vector
alpha=np.concatenate((alpha1*np.ones(N/2),alpha2*np.ones(N-N/2)))
#==============================================================================
#Load afferent input PSPs from Poisson spiking
#==============================================================================
xeps=np.load("inputspikes"+str(rate1)+str(rate2)+".npy")



#==============================================================================
# Sample test Poisson input spike trains from N afferents 
#==============================================================================

#initialize spike trains
xtest=np.ones((N,testtimesteps,testsample))

#calculate rate vectors for the two neuron groups
r1=rate1*np.ones(N/2)
r2=rate2*np.ones(N-N/2)
    
# rate vector for whole population
r=np.concatenate((r1,r2))
    
#sample Poisson spike train from rates
xtest[0:N/2,:,:]=np.random.binomial(1,rate1*dt,(N/2,testtimesteps,testsample))
xtest[N/2:N,:,:]=np.random.binomial(1,rate2*dt,(N-N/2,testtimesteps,testsample))

#positive and negative psp components
xepstests,xepstestl=np.zeros((2,N,testtimesteps,testsample))
	 
for time in np.arange(1,testtimesteps):      	
    xepstests[:,time,:]=(np.exp(-dt/taus)*xepstests[:,time-1,:]+xtest[:,time,:])
    xepstestl[:,time,:]=(xepstestl[:,time-1,:]*np.exp(-dt/taul)+xtest[:,time,:])
    
    #calculate psp    
    xepstest=(1./(taul-taus))*(xepstestl-xepstests)
    
    
#==============================================================================
# Run Euclidean gradient algorithm
#==============================================================================

#initialize weight
weight=np.copy(wstart)

#weightvec keeps track of weightpath
weightvec=np.zeros((N,l_intervals+1))
weightvec[:,0]=weight

#loop over learning intervals
for learn_interval in range(l_intervals):
	#sample indices of input spike train
    input_sample1=np.random.randint(0,sample_spiketrains,(N/2,))
    input_sample2=np.random.randint(sample_spiketrains,2*sample_spiketrains,(N-N/2,))
    input_sample=np.concatenate((input_sample1,input_sample2))
	

	#sample target spikes 
    vwopt=np.tile(np.reshape(wopt,(N,1,)),(1,timesteps,))
    Vtarget=np.einsum("ij,ij->j",vwopt,xeps[input_sample,:])
    yopt=np.random.binomial(1,maxfire*vphi(Vtarget)*dt,(timesteps,))

	#euclidean gradient learning
    for i in range(timesteps-1):
        #membrane potential
        V=np.dot(weight,xeps[input_sample,i])
    	#euclidean gradient update (Eqn. 10)
        weight+=-lre*(1.-phi(V))*(maxfire*dt*phi(V)-yopt[i])*xeps[input_sample,i]
    		
    weightvec[:,learn_interval+1]=weight
    

#initialize rate error, cost function, weight distance
error,cost,weightdist=np.zeros((3,l_intervals/interval+1))

#calculate target potential across testtimesteps and testsamples
vwopt=np.tile(np.reshape(wopt,(N,1,)),(1,testsample,))
Vtargettc=np.einsum("ij,ij->j",vwopt,xepstest[:,-1,:])

#calculate cost and error
for index in np.arange(0,l_intervals/interval+1):
    time=index*interval #not the real time but the index of the weight vector. For real time multiply by timesteps*dt
    
    vweight=np.tile(np.reshape(weightvec[:,time:time+1],(N,1)),(1,testsample))
    
    #caluclate somatic potential timecourse across testtimesteps and testsamples
    Vtc=np.einsum("ij,ij->j",vweight,xepstest[:,-1,:])
    
    #calculate error
    error[index]=maxfire*np.sqrt(np.mean((vphi(Vtc)-vphi(Vtargettc))**2))
    cost[index]=np.mean((np.log(vphi(Vtargettc))-np.log(vphi(Vtc)))*maxfire*vphi(Vtargettc)*dt +(1-maxfire*vphi(Vtargettc)*dt)*(np.log(1.-maxfire*dt*vphi(Vtargettc))-np.log(1.-maxfire*dt*vphi(Vtc)))) 
    weightdist[index]=np.sqrt(np.dot(np.reshape(weightvec[:,time:time+1],-1)-wopt,np.reshape(weightvec[:,time:time+1],-1)-wopt))
#save cost, error and weightpath, (disable if only one instance should be run manually for testing)   
np.save("2018-08-23_egradient_rate_error"+str(t)+str(lre)+".npy",error)
np.save("2018-08-23_egradient_KL_error"+str(t)+str(lre)+".npy",cost)
np.save("2018-08-23_ewpath_rate_error"+str(t)+str(lre)+".npy",weightvec[:,::interval])
np.save("2021-08-21_e_weightdist"+str(t)+str(lre)+".npy",weightdist)



#=========================================================================================
#lookup table
#=========================================================================================

q=1./intepsquad*np.sum(r)
                 
lookup_table=np.load(lookup_table_file)
#find right place in the lookup table for the different input rate configurations
qlist=np.array((5,7,26,39,52,78,260,390,520,780,2600,3900,7800))
qindex=np.where(qlist==int(q))[0][0]


#==============================================================================
# Natural gradient algorithm
#==============================================================================

#initialize weights
weight=np.copy(wstart)

#weightvec keeps track of weightpath
weightvec=np.zeros((N,l_intervals+1))
weightvec[:,0]=weight
#loop over learning intervals
for learn_interval in range(l_intervals):

	#sample indices of input spike trains
	input_sample1=np.random.randint(0,sample_spiketrains,(N/2,))
	input_sample2=np.random.randint(sample_spiketrains,2*sample_spiketrains,(N-N/2,))
	input_sample=np.concatenate((input_sample1,input_sample2))

	#sample target spikes
	vwopt=np.tile(np.reshape(wopt,(N,1,)),(1,timesteps,))
	Vtarget=np.einsum("ij,ij->j",vwopt,xeps[input_sample,:])
	yopt=np.random.binomial(1,maxfire*vphi(Vtarget)*dt,(timesteps,))


	for i in np.arange(0,timesteps-1):
    
    		#calculate m=Vmean and vtilde=Var V/intepsquad to look up coefficients in lookup table
            m=inteps*(np.dot(weight[0:(N/2)],r1)+np.dot(weight[(N/2):N],r2))
            var=intepsquad*(np.dot(np.square(weight[0:(N/2)]),r1)+np.dot(np.square(weight[(N/2):N]),r2))
    		#get gs from lookup table
            gcoeff=lookup_table[qindex,int(np.maximum(0.,np.minimum((m+15.)*4.,120.))),int(np.maximum(0.,np.minimum((var-1.)*2,819.))),:]	
    
    		#membrane potential
            V=np.dot(weight,xeps[input_sample,i])
            gamma0=gcoeff[0]  #Eqn. 107
            gamma_u=-1./intepsquad*(1./intepsquad*np.sum(xeps[input_sample,i])*gcoeff[1]+V*gcoeff[3])#Eqn.108
            gamma_w=(1./intepsquad*np.sum(xeps[input_sample,i])*gcoeff[2]+V*gcoeff[4])              #Eqn. 109
    		#calculate natural gradient weight update (Eqn. 13)
            weight+=lrn*(-gcoeff[0]*(1.-phi(V)) *(maxfire*dt*phi(V)-yopt[i])*(xeps[input_sample,i]/(intepsquad*r)-gamma_u*np.ones(N)+gamma_w*weight))           
    		
    #save weight trace
   	weightvec[:,learn_interval+1]=weight



#initialize cost and error
error,cost,weightdist=np.zeros((3,l_intervals/interval+1))

#calculate target potential across testtimesteps and testsamples
vwopt=np.tile(np.reshape(wopt,(N,1,)),(1,testsample,))
Vtargettc=np.einsum("ij,ij->j",vwopt,xepstest[:,-1,:])

#calculate cost and error
for index in np.arange(0,l_intervals/interval+1):
    time=index*interval #not the real time but the index of the weight vector. For real time multiply by timesteps*dt
    
    vweight=np.tile(np.reshape(weightvec[:,time:time+1],(N,1)),(1,testsample))
    
    #caluclate somatic potential timecourse across testtimesteps and testsamples
    Vtc=np.einsum("ij,ij->j",vweight,xepstest[:,-1,:])
    
    #calculate error
    error[index]=maxfire*np.sqrt(np.mean((vphi(Vtc)-vphi(Vtargettc))**2))
    cost[index]=np.mean((np.log(vphi(Vtargettc))-np.log(vphi(Vtc)))*maxfire*vphi(Vtargettc)*dt +(1-maxfire*vphi(Vtargettc)*dt)*(np.log(1.-maxfire*dt*vphi(Vtargettc))-np.log(1.-maxfire*dt*vphi(Vtc)))) 
    weightdist[index]=np.sqrt(np.dot(np.reshape(weightvec[:,time:time+1],-1)-wopt,np.reshape(weightvec[:,time:time+1],-1)-wopt))
    
#save error, cost and weightpath, #disable if only one instance should be run manually for testing
np.save("2018-08-23_ngradient_rate_error"+str(t)+str(lrn)+".npy",error)
np.save("2018-08-23_ngradient_KL_error"+str(t)+str(lrn)+".npy",cost)
np.save("2018-08-23_nwpath_rate_error"+str(t)+str(lrn)+".npy",weightvec[:,::interval])
np.save("2021-08-21_n_weightdist"+str(t)+str(lrn)+".npy",weightdist)


#==============================================================================
# Approximative natural gradient algorithm
#==============================================================================

#initialize weights
weight=np.copy(wstart)

#weightvec keeps track of weightpath
weightvec=np.zeros((N,l_intervals+1))
weightvec[:,0]=weight

#loop over learning intervals
for learn_interval in range(l_intervals):

	#sample indices of input spike trains
    input_sample1=np.random.randint(0,sample_spiketrains,(N/2,))
    input_sample2=np.random.randint(sample_spiketrains,2*sample_spiketrains,(N-N/2,))
    input_sample=np.concatenate((input_sample1,input_sample2))

	#sample target spikes
    vwopt=np.tile(np.reshape(wopt,(N,1,)),(1,timesteps,))
    Vtarget=np.einsum("ij,ij->j",vwopt,xeps[input_sample,:])
    yopt=np.random.binomial(1,maxfire*vphi(Vtarget)*dt,(timesteps,))
    gamma_u=0.95
    gamma_w=V*0.05 

    for i in np.arange(0,timesteps-1):
    
    		#calculate m=mean(V) and v=Variance (V) to look up coefficients in lookup table
            m=inteps*(np.dot(weight[0:(N/2)],r1)+np.dot(weight[(N/2):N],r2))
            var=intepsquad*(np.dot(np.square(weight[0:(N/2)]),r1)+np.dot(np.square(weight[(N/2):N]),r2))
    		#get gs from lookup table    
            gcoeff=lookup_table[qindex,int(np.maximum(0.,np.minimum((m+15.)*4.,120.))),int(np.maximum(0.,np.minimum((var-1.)*2,819.))),:]	
    		#membrane potential
            V=np.dot(weight,xeps[input_sample,i])
            #apply the approximated natural gradient learning rule (Eqn. 114)
            weight+=lra*(-gcoeff[0]*(1.-phi(V)) *(maxfire*dt*phi(V)-yopt[i])*(xeps[input_sample,i]/(intepsquad*r)-gamma_u/intepsquad*np.ones(N)+gamma_w*weight))           
    		
    #save weight trace
    weightvec[:,learn_interval+1]=weight


#initialize error function
error,cost,weightdist=np.zeros((3,l_intervals/interval+1))

#calculate target potential across testtimesteps and testsamples
vwopt=np.tile(np.reshape(wopt,(N,1,)),(1,testsample,))
Vtargettc=np.einsum("ij,ij->j",vwopt,xepstest[:,-1,:])

#calculate cost and error
for index in np.arange(0,l_intervals/interval+1):
	   time=index*interval #not the real time but the index of the weight vector. For real time multiply by timesteps*dt
    
	   vweight=np.tile(np.reshape(weightvec[:,time:time+1],(N,1)),(1,testsample))
    
       #caluclate somatic potential timecourse across testtimesteps and testsamples
   	   Vtc=np.einsum("ij,ij->j",vweight,xepstest[:,-1,:])
    
       #calculate error
 	   error[index]=maxfire*np.sqrt(np.mean((vphi(Vtc)-vphi(Vtargettc))**2))
	   cost[index]=np.mean((np.log(vphi(Vtargettc))-np.log(vphi(Vtc)))*maxfire*vphi(Vtargettc)*dt +(1-maxfire*vphi(Vtargettc)*dt)*(np.log(1.-maxfire*dt*vphi(Vtargettc))-np.log(1.-maxfire*dt*vphi(Vtc)))) 

#save error and weightpath, diable if only one instance should be run for testing
np.save("2018-08-23_approx_rate_error"+str(t)+str(lra)+".npy",error)
np.save("2018-08-23_approx_KL_error"+str(t)+str(lra)+".npy",cost)
np.save("2018-08-23_awpath_rate_error"+str(t)+str(lra)+".npy",weightvec[:,::interval])

