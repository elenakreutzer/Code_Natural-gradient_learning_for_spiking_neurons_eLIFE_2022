# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 14:55:39 2018

@author: kreutzer
"""
import numpy as np 
import scipy.integrate as spig



#Note: due to change of notation:f1...f4 in this script =g1,..g4 in Eq1. 102-105

parameter={
    "taul":0.01,
    "taus":0.003,
    "samplenumber_inputs":20,
    "samplenumber_weights":20,
    "N":100,
    "Ns":np.array((10.,50.,100.,200.,)),
    "wmin":-5.,
    "wmax":5.,
    "dt":0.0005,
    "timesteps":400,
    "fixedrate":20.,
    "rates":np.arange(5.,55.,10.),
    "inteps":1.,
    "intepsquad":38.46,
    "threshold":10.,
    "slope":0.3,
    "maxfire":100.,
    "ofile":"2018-02-14_gamma1_avrate_sp.svg",
    "xlabel":r"$\bar{r}$",
    "ylabel":r"$\gamma_1$"
}



###############################################################################
#sample uniform weights
###############################################################################
def weightsample(**para):
    N=para.get("N")
    samplenumber_weights=para.get("samplenumber_weights")
    wmin=para.get("wmin")
    wmax=para.get("wmax")
    weight=np.random.uniform(wmin,wmax,(N,samplenumber_weights))
    return(weight)

###############################################################################
#afferent spike train
###############################################################################

def inputs(rate,**para):
    timesteps=para.get("timesteps")
    samplenumber_inputs=para.get("samplenumber_inputs")
    N=para.get("N")
    dt=para.get("dt")
    taul=para.get("taul")
    taus=para.get("taus")
    ratevec=np.tile(np.reshape(rate,(N,1,1)),(1,timesteps,samplenumber_inputs))
    x=np.zeros((N,timesteps,samplenumber_inputs))
    x=np.random.binomial(1,ratevec*dt,(N,timesteps,samplenumber_inputs))   
    xeps_pos,xeps_neg,xeps=np.zeros((3,N,timesteps,samplenumber_inputs))
    
    for time in np.arange(timesteps):
        xeps_pos[:,time,:]=xeps_pos[:,time-1,:]*np.exp(-dt/taul)+x[:,time,:]
        xeps_neg[:,time,:]=xeps_neg[:,time-1,:]*np.exp(-dt/taus)+x[:,time,:]
    xeps=1./(taul-taus)*(xeps_pos-xeps_neg)
    return({"xeps":xeps[:,-1,:]})
    


###############################################################################
#define transfer function and its derivative
###############################################################################    
def phi(u,**para):
              threshold=para.get("threshold")#threshold=center phi -resting_potential
              slope=para.get("slope")
                            
              if u>=0.:
                  return(1./(1.+np.exp(-slope*(u-threshold))))
              else:
                 return(np.exp(slope*(u-threshold))/(1.+np.exp(slope*(u-threshold))))
                 
           
def phiprime(u,**para):
      slope=para.get("slope")
      return(slope*phi(u,**para)*(1.-phi(u,**para)))
      
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
        

        f1=K1*K2*(-1./K2*c1+(c1*c2t*m+(c2t**2)*v)-(K1)*c1*((c1*c2t*m+v*c2t**2)*q+(c1*c3t*m+c2t*c3t*v)*m))

        f2=K1*K2*(-c2t/K2+(c1*c3t*m+c2t*c3t*v)*(1.-K1*m*c2t)-K1*c2t*(c1*c2t*m+v*c2t**2)*q)

        f3=K1*K2*((c1*c2t*q+c1*c3t*m)-c2t/K1)

        f4=K1*K2*(((c2t**2)*q+c2t*c3t*m)-c3t/K1)
        coefficients={"gammas":1/c1,"f1":f1,"f2":f2,"f3":f3,"f4":f4}
        return(coefficients)
###############################################################################
#calculate samples from gamma_u
###############################################################################

def gammau(inputweight,rate,**para):
    
    N=para.get("N")
    samplenumber_inputs=para.get("samplenumber_inputs")
    inteps=para.get("inteps")
    intepsquad=para.get("intepsquad")
    c_eps=1./intepsquad
    weight=1./N*inputweight

    gu,s=np.zeros((2,samplenumber_inputs))
        
    q=1./intepsquad*np.sum(rate)
    m=inteps/N*np.dot(inputweight,rate)
    v=(1./N**2)*intepsquad *np.dot(np.square(inputweight),rate)    
    
    gcoeff=g(q,m,v,**para)
    xeps=inputs(rate,**para)["xeps"]
    vweight=np.tile(np.reshape(weight,(N,1)),(1,samplenumber_inputs))
    
    V=np.einsum("ij,ij->j",vweight,xeps)
    gu=-c_eps*(c_eps*np.sum(xeps,axis=0)*gcoeff["f1"]+V*gcoeff["f3"])
    s=np.sum(xeps,axis=0)
    approx=c_eps*s/np.sum(rate)
    
    return({"gu":gu,"approx":approx})
    
    
###############################################################################
#calculate samples from gamma_w
###############################################################################

def gammaw(inputweight,rate,cst,**para):
    
    N=para.get("N")
    samplenumber_inputs=para.get("samplenumber_inputs")
    inteps=para.get("inteps")
    intepsquad=para.get("intepsquad")
    c_eps=1./intepsquad
    weight=1./N*inputweight

    gw,s=np.zeros((2,samplenumber_inputs))
        
    q=1./intepsquad*np.sum(rate)
    m=inteps/N*np.dot(inputweight,rate)
    v=(1./N**2)*intepsquad *np.dot(np.square(inputweight),rate)    
    
    gcoeff=g(q,m,v,**para)
    xeps=inputs(rate,**para)["xeps"]
    vweight=np.tile(np.reshape(weight,(N,1)),(1,samplenumber_inputs))
    
    V=np.einsum("ij,ij->j",vweight,xeps)
    gw=(c_eps*np.sum(xeps,axis=0)*gcoeff["f2"]+V*gcoeff["f4"])
    approx=V*gcoeff["f4"]
    approx_cst=V*cst
    
    return({"gw":gw,"approx":approx,"approx_cst":approx_cst})

###############################################################################
#calculate samples from f1
###############################################################################    
    
def fs(inputweight,rate,**para):
    
    N=para.get("N")
    inteps=para.get("inteps")
    intepsquad=para.get("intepsquad")
    
           
    q=1./intepsquad*np.sum(rate)
    m=inteps/N*np.dot(inputweight,rate)
    v=(1./N**2)*intepsquad *np.dot(np.square(inputweight),rate)    
    
    gcoeff=g(q,m,v,**para)
    gcoeff['m']=m
    
    
    return(gcoeff)
   

###############################################################################
#sampling
###############################################################################
def sampling_gu(ofile_gu,ofile_approx, **para):
    N=para.get("N")
    rates=para.get("rates")
    samplenumber_weights=para.get("samplenumber_weights")
    samplenumber_inputs=para.get("samplenumber_inputs")
    
    wsample=weightsample(**para)
    gu,approx=np.zeros((2,np.shape(rates)[0],samplenumber_weights,samplenumber_inputs))
    
    for i in range(np.shape(rates)[0]):
        for ws in range(samplenumber_weights):
            sample=gammau(wsample[:,ws],rates[i]*np.ones(N),**para)
            gu[i,ws]=sample["gu"]
            approx[i,ws]=sample["approx"]
    np.save(ofile_gu,gu)
    np.save(ofile_approx,approx)
    

def sampling_gw(ofile_gw,ofile_approx,ofile_approx_cst,cst, **para):
    N=para.get("N")
    rates=para.get("rates")
    samplenumber_weights=para.get("samplenumber_weights")
    samplenumber_inputs=para.get("samplenumber_inputs")
    ofile_approx_cst=ofile_approx_cst+str(cst)
    
    wsample=weightsample(**para)
    gw,approx,approx_cst=np.zeros((3,np.shape(rates)[0],samplenumber_weights,samplenumber_inputs))
    
    for i in range(np.shape(rates)[0]):
        for ws in range(samplenumber_weights):
            sample=gammaw(wsample[:,ws],rates[i]*np.ones(N),cst,**para)
            gw[i,ws]=sample["gw"]
            approx[i,ws]=sample["approx"]
            approx_cst[i,ws]=sample["approx_cst"]
    np.save(ofile_gw,gw)
    np.save(ofile_approx,approx)
    np.save(ofile_approx_cst,approx_cst)
    

def sampling_fs(ofile_m,ofile_phi_m,ofile_gs,ofile_f1,ofile_f2,ofile_f3,ofile_f4,ofile_rates, **para):
    N=para.get("N")
    rates=para.get("rates")
    samplenumber_weights=para.get("samplenumber_weights")
    maxfire=para.get("maxfire")
    
   
    wsample=weightsample(**para)
    m,phi_m,gs,f1,f2,f3,f4,rate=np.zeros((8,np.shape(rates)[0],samplenumber_weights))
    
    for i in range(np.shape(rates)[0]):
        for ws in range(samplenumber_weights):
            sample=fs(wsample[:,ws],rates[i]*np.ones(N),**para)
            m[i,ws]=sample["m"]
            phi_m[i,ws]=maxfire*phi(m[i,ws],**para)
            gs[i,ws]=sample["gammas"]
            f1[i,ws]=sample["f1"]
            f2[i,ws]=sample["f2"]
            f3[i,ws]=sample["f3"]
            f4[i,ws]=sample["f4"]
            rate[i,ws]=rates[i]
    np.save(ofile_m,m)
    np.save(ofile_phi_m,phi_m)
    np.save(ofile_gs,gs)
    np.save(ofile_f1,f1)
    np.save(ofile_f2,f2)
    np.save(ofile_f3,f3)
    np.save(ofile_f4,f4)
    np.save(ofile_rates,rate)   
    
    
def sampling_fs_N(ofile_f1N,ofile_f2N,ofile_f3N,ofile_f4N,ofile_N, **para):
    Ns=para.get("Ns")
    samplenumber_weights=para.get("samplenumber_weights")
    rate=para.get("fixedrate")
    
   
    
    f1N,f2N,f3N,f4N,N=np.zeros((5,np.shape(Ns)[0],samplenumber_weights))
    
    for i in range(np.shape(Ns)[0]):
        para["N"]=int(Ns[i])
        wsample=weightsample(**para)
        for ws in range(samplenumber_weights):
            sample=fs(wsample[:,ws],rate*np.ones(int(Ns[i])),**para)
            f1N[i,ws]=sample["f1"]
            f2N[i,ws]=sample["f2"]
            f3N[i,ws]=sample["f3"]
            f4N[i,ws]=sample["f4"]
            N[i,ws]=Ns[i]
    
    np.save(ofile_f1N,f1N)
    np.save(ofile_f2N,f2N)
    np.save(ofile_f3N,f3N)
    np.save(ofile_f4N,f4N)
    np.save(ofile_N,N)   

def sampling_f1_Nr(ofile_f1,ofile_approx,ofile_N, **para):
    Ns=para.get("Ns")
    print(np.shape(Ns)[0])
    rates=para.get("rates")
    print(np.shape(rates)[0])
    samplenumber_weights=para.get("samplenumber_weights")
    intepsquad=para.get("intepsquad")
    f1,approx,Nr=np.zeros((3,np.shape(Ns)[0],np.shape(rates)[0],samplenumber_weights))
    
    for i in range(np.shape(Ns)[0]):
        para["N"]=int(Ns[i])
        wsample=weightsample(**para)
   
        for j in range(np.shape(rates)[0]):
            for ws in range(samplenumber_weights):
                sample=fs(wsample[:,ws],rates[j]*np.ones(int(Ns[i])),**para)
                f1[i,j,ws]=sample["f1"]
                approx[i,j,ws]=-intepsquad/(Ns[i]*rates[j])
                Nr[i,j,ws]=Ns[i]*rates[j]
    
    np.save(ofile_f1,f1)
    np.save(ofile_approx,approx)
    np.save(ofile_N,Nr) 

    
    
###############################################################################
#sample gammas
###############################################################################    
sampling_gu("2020-04-26_gammau_exact", "2020-04-26_gammau_approx",**parameter)
sampling_gw("2020-04-26_gammaw_exact", "2020-04-26_gammaw_approx","2020-04-26_gammaw_approx_cst",cst=0.04,**parameter)


###############################################################################
#sample fs
###############################################################################    
sampling_fs("2020-04-26_m","2020-04-26_phi_m","2020-04-26_gs","2020-04-26_f1", "2020-04-26_f2","2020-04-26_f3","2020-04-26_f4","2020-04-26_fs_rates",**parameter)
sampling_fs_N("2020-04-26_f1N", "2020-04-26_f2N","2020-04-26_f3N","2020-04-26_f4N","2020-04-26_fs_N",**parameter)
sampling_f1_Nr("2020-05-28_f1Nr","2020-05-28_f1approx","2020-05-28_N_Nr",**parameter)