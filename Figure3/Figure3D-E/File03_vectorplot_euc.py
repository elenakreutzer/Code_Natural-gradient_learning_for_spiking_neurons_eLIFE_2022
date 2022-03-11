# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 15:44:07 2017

@author: kreutzer
"""



###############################################################################
#import modules
###############################################################################
import numpy as np
import os 
import File01_parameter_vectorplot as parameter
para=parameter.parameter
t=os.environ["SLURM_ARRAY_TASK_ID"]

threshold=para.get("threshold")
slope=para.get("slope")
maxfire=para.get("maxfire")
wmin=para.get("wmin")
wmax=para.get("wmax")
N=para.get("N")
taul=para.get("taul")
taus=para.get("taus")
dt=para.get("dt")
rate1=para.get("rate1")
rate2=para.get("rate2")
samplenumber=para.get("samplenumber")
timesteps=para.get("timesteps")
target=para.get("target")
precision=para.get("precision")



###############################################################################
#define transfer function and derivative
###############################################################################

def phi(u):     
                            
  if u>=0.:
     return(1./(1.+np.exp(-slope*(u-threshold))))# careful! slope appears in f
  else:
     return(np.exp(slope*(u-threshold))/(1.+np.exp(slope*(u-threshold))))
                 
           
def phiprime(u):
      return(slope*phi(u)*(1.-phi(u)))
      



 
    
    
###############################################################################
#sample inputpattern and calculate psps
###############################################################################

x=np.zeros((N,timesteps))
    
    
    
x[0:N/2,:]=np.random.binomial(1,rate1*dt,(N/2,timesteps))
x[N/2:N,:]=np.random.binomial(1,rate2*dt,(N-N/2,timesteps))
    
xeps_pos=x[:,0]
xeps_neg=x[:,0]
    
for time in np.arange(1,timesteps):
            xeps_pos=xeps_pos*np.exp(-dt/taul)+x[:,time]
            xeps_neg=xeps_neg*np.exp(-dt/taus)+x[:,time]
        
    
        
xeps=1./(taul-taus)*(xeps_pos-xeps_neg)

                 
###################################################################################
#euclidean gradient vectorplot
###################################################################################


eucvec=np.zeros((int((wmax-wmin)/precision)+1, int((wmax-wmin)/precision)+1,N))     
for i,w1i in enumerate(np.arange(wmin,wmax,precision)):
    for j,w2j in enumerate(np.arange(wmin,wmax,precision)):

        weight=1./N*np.concatenate((w1i*np.ones(N/2),w2j*np.ones(N-N/2)))
        V=np.dot(weight,xeps)
        
        wopt=1./N*np.concatenate((target[0]*np.ones(N/2),target[1]*np.ones(N-N/2)))
        Vopt=np.dot(wopt,xeps)
               
        eucvec[i,j,:]=-phiprime(V)/phi(V)*(phi(V)-phi(Vopt))*xeps
  
np.save("2019-01-07_eucvec"+str(t),eucvec)      

        
        

                    







