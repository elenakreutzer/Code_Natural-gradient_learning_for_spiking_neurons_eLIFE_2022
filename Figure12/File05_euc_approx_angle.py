# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 15:44:07 2017

@author: kreutzer
"""


###############################################################################
#import modules
###############################################################################
import numpy as np                                                             #import numpy package
import os                                                                      #module needed to run multiple instances on SLURM cluster
import File01_parameter_angle as parameter                               #import file that contains all simulation parameters
t=os.environ["SLURM_ARRAY_TASK_ID"]                                            #run multiple instances on a slurm cluster, disable if only a single instance should be run manually

para=parameter.parameter                                                       #see comments in "File01_simulation_parameters" for explanations of the different parameters
threshold=para.get("threshold")
slope=para.get("slope")
maxfire=para.get("maxfire")
wmin=para.get("wmin")
wmax=para.get("wmax")
N=para.get("N")
taul=para.get("taul")
taus=para.get("taus")
intepsquad=para.get("intepsquad")
inteps=para.get("inteps")
dt=para.get("dt")
rate1=para.get("rate1")
rate2=para.get("rate2")
samplenumber=para.get("samplenumber")
timesteps=para.get("timesteps")
precision=para.get("precision")
r=np.concatenate((rate1*np.ones(N/2),rate2*np.ones(N-N/2)))
lookup_table_file=para.get("lt")

###############################################################################
#define transfer function and derivative
###############################################################################

def phi(u):     
                            
  if u>=0.:
     return(1./(1.+np.exp(-slope*(u-threshold))))# careful! slope appears in f
  else:
     return(np.exp(slope*(u-threshold))/(1.+np.exp(slope*(u-threshold))))
                 
           
def phip(u):
      return(slope*phi(u)*(1.-phi(u)))
      

    
###############################################################################
#sample inputpattern and calculate psps
###############################################################################

x=np.zeros((N,timesteps,samplenumber))
    
    
    
x[0:N/2,:,:]=np.random.binomial(1,rate1*dt,(N/2,timesteps,samplenumber))
x[N/2:N,:,:]=np.random.binomial(1,rate2*dt,(N-N/2,timesteps,samplenumber))
    
xeps_pos=x[:,0,:]
xeps_neg=x[:,0,:]
for time in np.arange(1,timesteps):
            xeps_pos=xeps_pos*np.exp(-dt/taul)+x[:,time,:]
            xeps_neg=xeps_neg*np.exp(-dt/taus)+x[:,time,:]
        

        
xeps=1./(taul-taus)*(xeps_pos-xeps_neg) 
                 



q=1./intepsquad*np.sum(r)
                 
lookup_table=np.load(lookup_table_file)

qlist=np.array((5,7,26,39,52,78,260,390,520,780,2600,3900,7800))
#qlist=np.array((5,26,52,260,520,2600))
qindex=np.where(qlist==int(q))[0][0]
        

        


###############################################################################
#calculate natural gradient vector at single point
###############################################################################        

weight=1./N*np.random.uniform(wmin,wmax,N)
wopt=1./N*np.random.uniform(wmin,wmax,N)

        
        
#calculate m=mean(V)and var=variance(V) to look up coefficients in lookup table

var=intepsquad*np.dot(np.square(weight),r)
m=inteps*(np.dot(weight,r))
angle, approx_angle,nf,nf_approx=np.zeros((4,samplenumber))

#get gs from lookup table
gcoeff=lookup_table[qindex,int(np.minimum((m+15.)*4.,120.)),int(np.minimum((var-1.)*2.,819.)),:]
for sample in range(samplenumber):	
    V=np.dot(weight,xeps[:,sample])
    Vopt=np.dot(wopt,xeps[:,sample])
    
    gamma_u=-1/intepsquad*(1/intepsquad*np.sum(xeps[:,sample])*gcoeff[1]+V*gcoeff[2]) #Eqn.108
    approxgamma_u=0.95/intepsquad                                                   #Eqn. 112
    gamma_w=(1/intepsquad*np.sum(xeps[:,sample])*gcoeff[3]+V*gcoeff[4])               #Eqn. 109
    approxgamma_w=0.05*V                                                              #Eqn. 113
	
    #calculate natural gradient and Euclidean gradient vector, and approximated natural gradient vector
    natvec=-gcoeff[0]*(phip(V)/phi(V)*(phi(V)-phi(Vopt))*((xeps[:,sample]/(intepsquad*r))-gamma_u*np.ones((N))+gamma_w*weight)) #Eqn.13
    eucvec=-phip(V)/phi(V)*(phi(V)-phi(Vopt))*xeps[:,sample]                                                                    #Eqn. 10
    approxvec=-gcoeff[0]*(phip(V)/phi(V)*(phi(V)-phi(Vopt))*((xeps[:,sample]/(intepsquad*r))-approxgamma_u*np.ones((N))+approxgamma_w*weight)) #Eqn.114
    
    #calculate norm of vectors
    norm_nat=np.sqrt(np.dot(natvec,natvec))        
    norm_euc=np.sqrt(np.dot(eucvec,eucvec))
    norm_approx=np.sqrt(np.dot(approxvec,approxvec))
    
    #calculate angles in Euclidean meatric
    angle[sample]=np.dot(natvec,eucvec)/(norm_nat*norm_euc)
    approx_angle[sample]=np.dot(natvec,approxvec)/(norm_nat*norm_approx)
	

#transform angles into degree    
angle=np.rad2deg(np.arccos(angle))
approx_angle=np.rad2deg(np.arccos(approx_angle))

#calculate mean angles
mean_angle=np.mean(angle)
mean_approx_angle=np.mean(approx_angle)

#save mean angles, disable if only one instance is run for testing (no cluster used)
np.save("2019-02-12_mean_euc_angle"+str(t)+str(N)+str(rate1)+str(rate2),mean_angle)             
np.save("2019-02-12_mean_euc_angle_approx"+str(t)+str(N)+str(rate1)+str(rate2),mean_approx_angle)             

#save mean angles, enable if only one instance is run for testing (no cluster used)
np.save("2019-02-12_mean_euc_angle_test"+str(N)+str(rate1)+str(rate2),mean_angle)             
np.save("2019-02-12_mean_euc_angle_approx_test"+str(N)+str(rate1)+str(rate2),mean_approx_angle)        
       
        
        
        
     
        
       
       
