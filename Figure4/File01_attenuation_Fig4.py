# -*- coding: utf-8 -*-
"""
Created on Fri May 25 10:34:38 2018

@author: kreutzer
"""
###############################################################################
#import libraries
###############################################################################
import numpy as np
import scipy.integrate as spig
import matplotlib.pyplot as plt


parameter={
    "initial_weight":0.05,
    "tau_m":0.01,
    "tau_s":0.003,
    "dt":0.0005,
    "intepsquad":38.46,   #integral over the squared input kernel, intepsquad=c_eps^{-1}
    "inteps":1.,
    "rate":5.,
    "lr":0.017,
    "threshold":10.,      #transfer function: rest=-70., center=-60.
    "slope":0.3,          #saturation around -50
    "maxfire":100.,       #max output firing rate
    "post_rate":20.,      #teacher Poisson firing rate
    "learning_time":5.,
    "invalpha":1.         #inverse attenuation factor
        }
###############################################################################
#define transfer function and derivative
###############################################################################

def phi(u,**para):
              threshold=para.get("threshold")#threshold=center sigmoidal -resting_potential
              slope=para.get("slope")
                            
              if u>=0.:
                  return(1./(1.+np.exp(-slope*(u-threshold))))
              else:
                 return(np.exp(slope*(u-threshold))/(1.+np.exp(slope*(u-threshold))))
                 
           
def phip(u,**para):
      slope=para.get("slope")
      return(slope*phi(u,**para)*(1.-phi(u,**para)))
  

###############################################################################
#sample input spike trains
###############################################################################
def init(**para):
    
    tau_m=para.get("tau_m")
    tau_s=para.get("tau_s")
    rate=para.get("rate")
    dt=para.get("dt")
    timesteps=int(para.get("learning_time")/dt)
    
    x=np.random.binomial(1,rate*dt,(timesteps,))
    xeps,xeps_pos,xeps_neg,post=np.zeros((4,timesteps))
    
    xeps_pos[0]=x[0]
    xeps_neg[0]=x[0]
            
    for time in np.arange(1,timesteps):
            xeps_pos[time]=xeps_pos[time-1]*np.exp(-dt/tau_m)+x[time]
            xeps_neg[time]=xeps_neg[time-1]*np.exp(-dt/tau_s)+x[time]
    
    xeps=1./(tau_m-tau_s)*(xeps_pos-xeps_neg)           
    
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
        K1=1./(I1*(q+1)+c2*m)                               #Eqn. 88, note k1=c1^{-1}K1
        K2=1./(I1+(c2*m+c3*v)-K1*(I1*m+c2*v)*(c2*q+c3*m))   #Eqn. 90, note k2=c1*K2
        

        g1=K1*K2*(-1./K2*I1+(I1*c2*m+(c2**2)*v)-(K1)*I1*((I1*c2*m+v*c2**2)*q+(I1*c3*m+c2*c3*v)*m)) #Eqn. 102

        g2=K1*K2*(-c2/K2+(I1*c3*m+c2*c3*v)*(1.-K1*m*c2)-K1*c2*(I1*c2*m+v*c2**2)*q)                 #Eqn. 103

        g3=K1*K2*((I1*c2*q+I1*c3*m)-c2/K1)                                                         #Eqn. 104 

        g4=K1*K2*(((c2**2)*q+c2*c3*m)-c3/K1)                                                       #Eqn. 105
        coefficients={"gamma0":1/I1,"g1":g1,"g2":g2,"g3":g3,"g4":g4}
        return(coefficients)
        






##############################################################################
#learn for fixed time period
###############################################################################

def nlearning(weight,**para):

    dt=para.get("dt")
    timesteps=int(para.get("learning_time")/dt)
    rate=para.get("rate")
    post_rate=para.get("post_rate")
    dt=para.get("dt")
    inteps=para.get("inteps")
    intepsquad=para.get("intepsquad")
    lr=para.get("lr")
    maxfire=para.get("maxfire")
    invalpha=para.get("invalpha")
    alpha=(invalpha)**(-1)
    

    w=np.zeros((timesteps,))
 
    w[0]=weight
    np.random.seed(55)

    q=rate/intepsquad
    wsom=(alpha)*weight       #somatic weight=attenuated dendritic weight
    m=inteps*wsom*rate        #mean membrane potential
    v=intepsquad*wsom**2*rate #variance of membrane potential
    gcoeff=g(q,m,v,**para)
    post=np.random.binomial(1,post_rate*dt,(timesteps,)) #sample teacher spikes
        
#learning
    xeps=init(**para)
    for timestep in range(timesteps-1):
        Vsom=wsom*xeps[timestep]
        gamma_u=-(1/intepsquad*xeps[timestep]*gcoeff["g1"]+wsom*xeps[timestep]*gcoeff["g3"])/intepsquad
        gamma_w=(1/intepsquad*xeps[timestep]*gcoeff["g2"]+wsom*xeps[timestep]*gcoeff["g4"])
        
        #Eqn.17
        w[timestep+1]=w[timestep]+lr*(post[timestep]-maxfire*phi(Vsom,**para)*dt)\
        *phip(Vsom,**para)/phi(Vsom,**para)\
        *gcoeff["gamma0"]*(invalpha*xeps[timestep]/(rate*intepsquad)-invalpha*gamma_u+gamma_w*w[timestep])#
        
    return(w)

#Data for Fig. 4C

deltaw,deltaw_abs,invalpha=np.zeros((3,10))

for index in range(10):
    invalpha[index]=index+1.
    parameter["invalpha"]=invalpha[index]
    w=nlearning((invalpha[index])*parameter["initial_weight"],**parameter)
    print(w)
    deltaw_abs[index]=w[-1]-w[0]
    deltaw[index]=w[-1]/w[0]
np.save("invalpha",invalpha)
np.save("att_weight_change_perc",100.*(deltaw-np.ones(10)))
np.save("att_weight_change_abs",deltaw_abs)
np.save("time_parameters",{"learning_time":parameter["learning_time"],"dt":parameter["dt"]})  
fig=plt.figure(1)
plt.scatter(invalpha,deltaw_abs)

deltaw_per=100.*(deltaw-np.ones(10))
fig2=plt.figure(2)
plt.scatter(invalpha,deltaw_per)

#Data for Fig. 4B

parameter["invalpha"]=3.
pweights=nlearning((parameter["invalpha"])*parameter["initial_weight"],**parameter)
np.save("attenuation_proximal_synapse",pweights)
fig3=plt.figure(3)
plt.plot(pweights)


parameter["invalpha"]=7.
dweights=nlearning((parameter["invalpha"])*parameter["initial_weight"],**parameter)
np.save("attenuation_distal_synapse",dweights)
plt.plot(dweights)
