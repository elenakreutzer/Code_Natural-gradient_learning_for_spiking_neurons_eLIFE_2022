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
intepsquad=para.get("intepsquad")
inteps=para.get("inteps")
dt=para.get("dt")
rate1=para.get("rate1")
rate2=para.get("rate2")
samplenumber=para.get("samplenumber")
timesteps=para.get("timesteps")
target=para.get("target")
precision=para.get("precision")
r=np.concatenate((rate1*np.ones(N/2),rate2*np.ones(N-N/2)))
lt_file=para.get("lt")

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
                 



q=1./intepsquad*np.sum(r)
                 
lt=np.load(lt_file)
        

        


###############################################################################
#calculate natural gradient vector at single point
###############################################################################        
natvec=np.zeros((int((wmax-wmin)/precision)+1,int((wmax-wmin)/precision)+1,N))     
for i,w1i in enumerate(np.arange(wmin,wmax,precision)):
    for j,w2j in enumerate(np.arange(wmin,wmax,precision)):

        weight=1./N*np.concatenate((w1i*np.ones(N/2),w2j*np.ones(N-N/2)))
        V=np.dot(weight,xeps)
        
        wopt=1./N*np.concatenate((target[0]*np.ones(N/2),target[1]*np.ones(N-N/2)))
        Vopt=np.dot(wopt,xeps)
        
        
    	#calculate m=Vmean and vtilde=Var V/intepsquad to look up coefficients in lookup table

        var=intepsquad*np.dot(np.square(weight),r)
        m=inteps*(np.dot(weight,r))
    	#get fs from lookup table
        gcoeff=lt[int(np.maximum(0.,np.minimum((m+15.)*4.,120.))),int(np.maximum(0.,np.minimum((var-1.)*2,819.))),:]	
        gamma1=-(1/intepsquad*np.sum(xeps)*gcoeff[1]+V*gcoeff[2])
        gamma2=(1/intepsquad*np.sum(xeps)*gcoeff[3]+V*gcoeff[4])
        
        natvec[i,j,:]=-gcoeff[0]*(phiprime(V)/phi(V)*(phi(V)-phi(Vopt))*((xeps/(intepsquad*r))-gamma1/intepsquad*np.ones((N))+gamma2*weight))*dt
        
np.save("2019-01-07_natvec"+str(t),natvec)             

        
        
        
        
        
        
     
        
       
       
