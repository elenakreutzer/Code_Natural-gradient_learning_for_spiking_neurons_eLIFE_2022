# -*- coding: utf-8 -*-
"""
Created on Fri May 25 10:34:38 2018

@author: kreutzer

"""


###############################################################################
#import modules
###############################################################################
import numpy as np
import scipy.integrate as spig
import os
from File01_weightrange_Fig7 import wr
t=os.environ['SLURM_ARRAY_TASK_ID']


#output files
ofile_hom="Delta_w_hom"+str(t)+".npy"
ofile_het="Delta_w_het"+str(t)+".npy"

#weights
w_min=wr["w_min"]
w_max=wr["w_max"]
steps=wr["steps"]


#######################################################################################################################################################################################
#parameter dictionary
######################################################################################################################################################################################

parameter={

"Ntot":10,#total number of neurons
"Nstim":5,#number of stimulated neurons
"taul":0.01,
"taus":0.003,
"dt":0.0005,
"intepsquad":38.46,
"rate":5,
"epsilon":0.0001,
"lr":0.001,
"threshold":15,#rest=-70, center=-60 #tonic inhibition=-5
"slope":0.3,#sigmoidal should become flat around -55
"maxfire":100.,
"post_rate":20,
"interval_length":60.,
}

##########################################################################################################################################################################################
#initialize weights
##########################################################################################################################################################################################
w=np.linspace(w_min,w_max,steps)
w_hom_change,w_het_change,w_hom_abs,w_het_abs=np.zeros((4,len(w)))
w_hom=w_min+float(t)*(w_max-w_min)/steps
w_stim=1./parameter["Ntot"]*w_hom*np.ones((parameter["Nstim"])) #paralleliez: for each stimulated weight one job on slurm cluster
###############################################################################
#define transfer function and derivative
###############################################################################

def phi(u,**para):
              threshold=para.get("threshold")#threshold=center phi -resting_potential
              slope=para.get("slope")
                            
              if u>=0.:
                  return(1./(1.+np.exp(-slope*(u-threshold))))# careful! slope appears in f
              else:
                 return(np.exp(slope*(u-threshold))/(1.+np.exp(slope*(u-threshold))))
                 
           
def phiprime(u,**para):
      slope=para.get("slope")
      return(slope*phi(u,**para)*(1.-phi(u,**para)))
      
      

###############################################################################
#set input- and output spike train according to pairing protocol
###############################################################################
def init(**para):
    Nstim=para.get("Nstim")
    taul=para.get("taul")
    taus=para.get("taus")
    rate=para.get("rate")
    dt=para.get("dt")
    timesteps=int(para.get("interval_length")/dt)
    
    x=np.random.binomial(1,rate*dt,(Nstim,timesteps,))
    xeps,xeps_pos,xeps_neg,post=np.zeros((4,Nstim,timesteps))
    
    xeps_pos[:,0]=x[:,0]
    xeps_neg[:,0]=x[:,0]
            
    for time in np.arange(1,timesteps):
            xeps_pos[:,time]=xeps_pos[:,time-1]*np.exp(-dt/taul)+x[:,time]
            xeps_neg[:,time]=xeps_neg[:,time-1]*np.exp(-dt/taus)+x[:,time]
            
    xeps=1./(taul-taus)*(xeps_pos-xeps_neg)           
    return(xeps)




###############################################################################
#calculate coefficients for learning rule
###############################################################################
def g(q,m,v,**para):             
        maxfire=para.get("maxfire")  
        slope=para.get("slope")

        I1=(1./(np.sqrt(2.*np.pi*v)))*spig.quad(lambda u: maxfire*(slope**2)*phi(u,**para)*((1.-phi(u,**para))**2)*np.exp(-0.5*((u-m)**2)/v),-np.inf,np.inf)[0]
        I2=(1./(np.sqrt(2.*np.pi*v)))*spig.quad(lambda u: maxfire*(slope**2)*phi(u,**para)*((1.-phi(u,**para))**2)*u*np.exp(-0.5*((u-m)**2)/v),-np.inf,np.inf)[0]
        I3=(1./(np.sqrt(2.*np.pi*v)))*spig.quad(lambda u: maxfire*(slope**2)*phi(u,**para)*((1.-phi(u,**para))**2)*(u**2)*np.exp(-0.5*((u-m)**2)/v),-np.inf,np.inf)[0]
        c2=(I2-I1*m)/(v)
        c3=(I3-I1*((m**2)+v)-2.*c2*m*v)/(v**2)
        K1=1./(I1*(q+1)+c2*m)
        K2=1./(I1+(c2*m+c3*v)-K1*(I1*m+c2*v)*(c2*q+c3*m))
        

        g1=K1*K2*(-1./K2*I1+(I1*c2*m+(c2**2)*v)-(K1)*I1*((I1*c2*m+v*c2**2)*q+(I1*c3*m+c2*c3*v)*m))

        g2=K1*K2*(-c2/K2+(I1*c3*m+c2*c3*v)*(1.-K1*m*c2)-K1*c2*(I1*c2*m+v*c2**2)*q)

        g3=K1*K2*((I1*c2*q+I1*c3*m)-c2/K1)

        g4=K1*K2*(((c2**2)*q+c2*c3*m)-c3/K1)
        coefficients={"gamma_s":1/I1,"g1":g1,"g2":g2,"g3":g3,"g4":g4}
        return(coefficients)
        
##################################
#learning rule
##################################        

def ngradient(inputweight,usp,**para):
    Nstim=para.get("Nstim")
    Ntot=para.get("Ntot")
    rate=para.get("rate")
    epsilon=para.get("epsilon")
    post_rate=para.get("post_rate")
    dt=para.get("dt")
    intepsquad=para.get("intepsquad")
    timesteps=int(para.get("interval_length")/dt)
    lr=para.get("lr")
    maxfire=para.get("maxfire")
    

    weightstim=np.copy(inputweight["stim"])
    weightunstim=np.copy(inputweight["unstim"])
        
    q=(Nstim*rate+(Ntot-Nstim)*epsilon)/intepsquad  
    post=np.random.binomial(1,post_rate*dt,(timesteps,))  
    
    m=(np.sum(weightstim)*rate+np.sum(weightunstim)*epsilon)
    v=intepsquad*(np.sum(np.square(weightstim))*rate+np.sum(np.square(weightunstim))*epsilon)
    gcoeff=g(q,m,v,**para)   
    
    for timestep in range(timesteps-1):
        
        #membrane potential and total input

        V=np.dot(weightstim,usp[:,timestep])
        xtot=np.sum(usp[:,timestep])
    
        #coefficients
        gamma_u=-(1/intepsquad*xtot*gcoeff["g1"]+V*gcoeff["g3"])/intepsquad
        gamma_w=(1/intepsquad*xtot*gcoeff["g2"]+V*gcoeff["g4"])
        
        #update stimulated synapses
        weightstim+=(lr*(post[timestep]-maxfire*phi(V,**para)*dt)*phiprime(V,**para)/phi(V,**para)\
        *gcoeff["gamma_s"]*(usp[:,timestep]/(rate*intepsquad)-gamma_u*np.ones(Nstim)+gamma_w*weightstim))
        
        #update unstimulated synapses
        weightunstim+= (lr*(post[timestep]-maxfire*phi(V,**para)*dt)*phiprime(V,**para)/phi(V,**para)\
        *gcoeff["gamma_s"]*(-gamma_u*np.ones(Ntot-Nstim)+gamma_w*weightunstim))
        
    return({"stim":weightstim,"unstim": weightunstim})


        

###############################################################################
#looping over unstimulated weights
###############################################################################
for i in range(len(w)):
	w_unstim=1./parameter["Ntot"]*w[i]*np.ones((parameter["Ntot"]-parameter["Nstim"]))
	usp=init(**parameter)
	w_change=ngradient({"unstim":np.copy(w_unstim),"stim":np.copy(w_stim)},usp,**parameter)
	w_hom_abs[i]=(parameter["Ntot"]*np.mean(w_change["stim"]))-w_hom#percentage weight change
	w_het_abs[i]=(parameter["Ntot"]*np.mean(w_change["unstim"]))-w[i]
np.save(ofile_hom,w_hom_abs)
np.save(ofile_het,w_het_abs)

np.save("hom"+str(t),w_stim)


    
    
