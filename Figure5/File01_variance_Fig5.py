# -*- coding: utf-8 -*-
"""
Created on Fri May 25 10:34:38 2018

@author: kreutzer

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spig

###############################################################################
#parameter
###############################################################################
parameter={
    "taul":0.01,
    "dt":0.0005,
    "lr":0.0001,
    "threshold":10.,        #rest=-70, #center=-60 saturation around -50
    "slope":0.3,
    "maxfire":100.,
    "post_rate":80.,        #Poisson firing rate of teacher
    "learning_time":5.,
    "initial_weight":.01,
    "eps_0":1.
     }

np.save("time_params",parameter)

###############################################################################
#define transfer function and derivative
###############################################################################

def phi(u,**para):
              threshold=para.get("threshold")#threshold=center sigmoidal -resting_potential
              slope=para.get("slope")
                            
              if u>=0.:
                  return(1./(1.+np.exp(-slope*(u-threshold))))# careful! slope appears in g
              else:
                 return(np.exp(slope*(u-threshold))/(1.+np.exp(slope*(u-threshold))))
                 
           
def phiprime(u,**para):
      slope=para.get("slope")
      return(slope*phi(u,**para)*(1.-phi(u,**para)))

###############################################################################
#sample input spike train
###############################################################################
def init(**para):
    taul=para.get("taul")
    taus=para.get("taus")
    rate=para.get("rate")
    dt=para.get("dt")
    eps_0=para.get("eps_0")
    timesteps=int(para.get("learning_time")/dt)
    
    x=np.random.binomial(1,rate*dt,(timesteps,))
    xeps,xeps_pos,xeps_neg,post=np.zeros((4,timesteps))
    
    xeps_pos[0]=x[0]
    xeps_neg[0]=x[0]
            
    for time in np.arange(1,timesteps):
            xeps_pos[time]=xeps_pos[time-1]*np.exp(-dt/taul)+x[time]
            xeps_neg[time]=xeps_neg[time-1]*np.exp(-dt/taus)+x[time]
    
    xeps=eps_0/(taul-taus)*(xeps_pos-xeps_neg)           
    return(xeps)




###############################################################################
#calculate coefficients for learning rule
###############################################################################
def g(q,m,v,**para):             
        maxfire=para.get("maxfire")  
        slope=para.get("slope")
        
        c1=(1./(np.sqrt(2.*np.pi*v)))*spig.quad(lambda u: maxfire*(slope**2)*phi(u,**para)*((1.-phi(u,**para))**2)*np.exp(-0.5*((u-m)**2)/v),-np.inf,np.inf)[0]
        c2=(1./(np.sqrt(2.*np.pi*v)))*spig.quad(lambda u: maxfire*(slope**2)*phi(u,**para)*((1.-phi(u,**para))**2)*u*np.exp(-0.5*((u-m)**2)/v),-np.inf,np.inf)[0]
        c3=(1./(np.sqrt(2.*np.pi*v)))*spig.quad(lambda u: maxfire*(slope**2)*phi(u,**para)*((1.-phi(u,**para))**2)*(u**2)*np.exp(-0.5*((u-m)**2)/v),-np.inf,np.inf)[0]
        c2t=(c2-c1*m)/(v)
        c3t=(c3-c1*((m**2)+v)-2.*c2t*m*v)/(v**2)
        K1=1./(c1*(q+1)+c2t*m)
        K2=1./(c1+(c2t*m+c3t*v)-K1*(c1*m+c2t*v)*(c2t*q+c3t*m))
        

        g1=K1*K2*(-1./K2*c1+(c1*c2t*m+(c2t**2)*v)-(K1)*c1*((c1*c2t*m+v*c2t**2)*q+(c1*c3t*m+c2t*c3t*v)*m))

        g2=K1*K2*(-c2t/K2+(c1*c3t*m+c2t*c3t*v)*(1.-K1*m*c2t)-K1*c2t*(c1*c2t*m+v*c2t**2)*q)

        g3=K1*K2*((c1*c2t*q+c1*c3t*m)-c2t/K1)

        g4=K1*K2*(((c2t**2)*q+c2t*c3t*m)-c3t/K1)
        coefficients={"gamma_s":1/c1,"g1":g1,"g2":g2,"g3":g3,"g4":g4}
        return(coefficients)
        


###############################################################################
#learning rule
###############################################################################
def ngradient(**para):
    np.random.seed(999)
    
    learning_time=para.get("learning_time")
    dt=para.get("dt")   
    timesteps=int(learning_time/dt)
    taul=para.get("taul")
    taus=para.get("taus")
    eps_0=para.get("eps_0")
    
    rate=para.get("rate")
    post_rate=para.get("post_rate")
    fac=(1./((taul-taus)))
    intepsquad=(fac**2)*(taul/2-(2/(1/taul+1/taus))+taus/2)

    lr=para.get("lr")
    maxfire=para.get("maxfire")
    
    usp=init(**para)  
    weight=np.zeros(timesteps,)
  
    weight[0]=para["initial_weight"]
        
    q=rate/intepsquad
    m=np.sum(weight)*eps_0*rate
    v=(eps_0**2)*intepsquad*(np.sum(np.square(weight))*rate)
    gcoeff=g(q,m,v,**para)
    
        
    post=np.random.binomial(1,post_rate*dt,(timesteps,)) #teacher spike train
    
    
        
    for timestep in range(timesteps-1):
        
        #membrane potential and total input
        V=np.dot(weight[timestep],usp[timestep])
        xtot=np.sum(usp[timestep])
    
        #coefficients
        gamma_u=-(1./(eps_0*intepsquad)*xtot*gcoeff["g1"]+V*gcoeff["g3"])*1./(eps_0*intepsquad)
        gamma_w=(1./(eps_0*intepsquad)*xtot*gcoeff["g2"]+V*gcoeff["g4"])
        
        #update stimulated synapses
        weight[timestep+1]=weight[timestep]+lr*((post[timestep]-maxfire*phi(V,**para)*dt)*phiprime(V,**para)/phi(V,**para)\
        *gcoeff["gamma_s"]*(usp[timestep]/(eps_0**2*intepsquad*rate)-gamma_u+gamma_w*weight[timestep]))
        
        
        
    return(weight,usp)

################################################################################
#histogram plot
###########################################################################
    
def plot_hist(ax,data_usp,color):
    
    ax.set_ylim([0.,5.])
    
         
    counts,bins,patches=ax.hist(data_usp,bins=np.arange(0.,10.,.5),log=0,orientation="horizontal",density=0.,rwidth=1.,color=color)
    ax.set_xlim([0.,np.sum(counts)*0.6])


    ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_xticks([0.,np.sum(counts)/4.])
    xtickslabel=([0,0.25,])
    ax.set_xticklabels(xtickslabel)
   
    ax.spines['left'].set_position(('axes', 1.))
    ax.tick_params(axis="y",direction="in", pad=-20.)
    ax.tick_params(axis="x",direction="out", pad=-0.3)
   
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    
    ax.set_xlabel(r'$p(x^{\epsilon})$',labelpad=0.)
    
       
    pass   


#######################################################################
#usp trace plot
######################################################################
    
def plot_usp_trace(ax,data,color,**para):
    ax.set_ylim([0.,5.])
    ax.set_xlim([0.,5.])
    

    ax.text(5.3,5.1,r"$r_2=50$ Hz",fontsize=8.)
    ax.text(5.3,4.1,r"$\tau_2=1$ ms",fontsize=8.)
    ax.text(5.3,3.1,r"$\sigma_2\approx 0.94$ mV",fontsize=8.)
    

    time=np.arange(0,para["learning_time"],para["dt"])
    
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    ax.plot(time,data,color=color,linewidth=.2,clip_on=False)
    
    ax.set_xlabel(r'$t$ [s]',labelpad=-2.)
    ax.set_ylabel(r'$x^{\epsilon}$ [mV]')

    pass
######################################################
#high rate high taus
######################################################    
parameter["taus"]=0.02
parameter["rate"]=50.
parameter["eps_0"]=1./parameter["rate"]

hhweights,hhusp=ngradient(**parameter)
plt.figure(0)
plt.plot(hhweights/parameter["initial_weight"]*100., color="springgreen")
np.save("variance_hr_ht",hhweights)
np.save("usp_trace_hr_ht",hhusp)

plt.figure(1)
ax=plt.axes()
plot_hist(ax,hhusp,"springgreen")

plt.figure(2)
ax=plt.axes()
plot_usp_trace(ax,hhusp,"springgreen",**parameter)

######################################################
#high rate low taus
######################################################    
parameter["taus"]=0.001
parameter["rate"]=50.
parameter["eps_0"]=1./parameter["rate"]

hlweights,hlusp=ngradient(**parameter)
plt.figure(0)
plt.plot(hlweights/parameter["initial_weight"]*100.,color="dodgerblue")
np.save("variance_hr_lt",hlweights)
np.save("usp_trace_hr_lt",hlusp)

plt.figure(3)
ax=plt.axes()
plot_hist(ax,hlusp,"dodgerblue")

plt.figure(4)
ax=plt.axes()
plot_usp_trace(ax,hlusp,"dodgerblue",**parameter)



######################################################
#low rate high taus
######################################################    
parameter["taus"]=0.02
parameter["rate"]=10.
parameter["eps_0"]=1./parameter["rate"]
lhweights,lhusp=ngradient(**parameter)
plt.figure(0)
plt.plot(lhweights/parameter["initial_weight"]*100.,color="darkorange")
np.save("variance_lr_ht",lhweights)
np.save("usp_trace_lr_ht",lhusp)

plt.figure(5)
ax=plt.axes()
plot_hist(ax,lhusp,"darkorange")
plt.figure(6)
ax=plt.axes()
plot_usp_trace(ax,lhusp,"darkorange",**parameter)

######################################################
#summary plot varied tau
######################################################
parameter["rate"]=10.
parameter["eps_0"]=1./parameter["rate"]
wfinal=np.zeros(6)
usp_variance=np.zeros(6)
tau=(0.001,0.003,0.005,0.008,0.015,0.02)
for i,item in enumerate(tau):
    taus=item
    parameter["taus"]=taus
    fac=(1./((parameter["taul"]-parameter["taus"])))
    intepsquad=(fac**2)*(parameter["taul"]/2-(2/(1/parameter["taul"]+1/parameter["taus"]))+parameter["taus"]/2)
    usp_variance[i]=parameter["eps_0"]**2*intepsquad*parameter["rate"]
    wfinal[i]=ngradient(**parameter)[0][-1:]
wfinal=(wfinal/parameter["initial_weight"]-1.)*100.
plt.figure(7)
plt.scatter(np.sqrt(usp_variance),wfinal, color="black")
np.save("usp_variance_tvar", usp_variance)
np.save("wfinal_tvar",wfinal)
np.save("taus",tau)



######################################################
#summary plot varied rate
######################################################      

parameter["taus"]=0.02

wfinal=np.zeros(5)
usp_variance=np.zeros(5)
rates=(10.,20.,30.,40.,50.)
for i,item in enumerate(rates):
    rate=item
    parameter["rate"]=rate
    parameter["eps_0"]=1./parameter["rate"]
    fac=(1./((parameter["taul"]-parameter["taus"])))
    intepsquad=(fac**2)*(parameter["taul"]/2-(2/(1/parameter["taul"]+1/parameter["taus"]))+parameter["taus"]/2)
    usp_variance[i]=parameter["eps_0"]**2*intepsquad*parameter["rate"]
    wfinal[i]=ngradient(**parameter)[0][-1:]
wfinal=(wfinal/parameter["initial_weight"]-1.)*100.
plt.figure(7)
plt.scatter(np.sqrt(usp_variance),wfinal,color="black")
np.save("usp_variance_rvar", usp_variance)
np.save("wfinal_rvar",wfinal)
np.save("rates",rates)





