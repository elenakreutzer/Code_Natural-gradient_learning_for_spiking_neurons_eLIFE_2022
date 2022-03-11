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
import File01_parameter_angle as parameter                                            #import file that contains all simulation parameters
t=os.environ["SLURM_ARRAY_TASK_ID"]                                            #run multiple instances on a slurm cluster

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
lt_file=para.get("lt")
ltc_file=para.get("ltc")

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
print(xeps, np.shape(xeps))                 



q=1./intepsquad*np.sum(r)
                 
lt=np.load(lt_file)
ltc=np.load(ltc_file)

qlist=np.array((5,7,26,39,52,78,260,390,520,780,2600,3900,7800))
qindex=np.where(qlist==int(q))[0][0]
        

        


###############################################################################
#calculate natural gradient vector at single point
###############################################################################        

weight=1./N*np.random.uniform(wmin,wmax,N)
wopt=1./N*np.random.uniform(wmin,wmax,N)
#save  weight
np.save("2019-02-12_wstart"+str(t)+".npy",weight)
np.save("2019-02-12_wopt"+str(t)+".npy",wopt)
        
     
        
#calculate m=Vmean and vtilde=Var V/intepsquad to look up coefficients in lookup table

var=intepsquad*np.dot(np.square(weight),r)
m=inteps*(np.dot(weight,r))
angle, approx_angle,nf,nf_approx=np.zeros((4,samplenumber))

#get gs from lookup table
gcoeff=lt[qindex,int(np.maximum(0.,np.minimum((m+15.)*4.,120.))),int(np.maximum(0.,np.minimum((var-1.)*2.,819.))),:]
ccoeff=ltc[qindex,int(np.maximum(0.,np.minimum((m+15.)*4.,120.))),int(np.maximum(0.,np.minimum((var-1.)*2.,819.))),:]

G=ccoeff[0]*(np.outer(r,r)+intepsquad*np.diag(r))+ccoeff[1]*intepsquad*((np.outer(np.dot(np.diag(r),weight),r)+np.outer(r,np.dot(np.diag(r),weight))))+ccoeff[2]*intepsquad**2*np.outer(np.dot(np.diag(r),weight),np.dot(np.diag(r),weight))
#invG=gcoeff[0]*(np.diag(1./(intepsquad*r))+np.outer((gcoeff[1]/intepsquad*np.ones(N)+gcoeff[2]*weight),np.ones(N)/intepsquad)+np.outer((gcoeff[3]/intepsquad*np.ones(N)+gcoeff[4]*weight),weight))
for sample in range(samplenumber):	
	V=np.dot(weight,xeps[:,sample])
	Vopt=np.dot(wopt,xeps[:,sample])
	gamma1=-(1/intepsquad*np.sum(xeps[:,sample])*gcoeff[1]+V*gcoeff[2])
	#approxgamma1=-(1/intepsquad*np.sum(xeps[:,sample])*gcoeff[1])
	#approxgamma1=np.sum(xeps[:,sample])/np.sum(r)
	approxgamma1=0.95
	
	gamma2=(1/intepsquad*np.sum(xeps[:,sample])*gcoeff[3]+V*gcoeff[4])
        approxgamma2=V*0.05
	#approxgamma2=gamma2

	natvec=-gcoeff[0]*(phip(V)/phi(V)*(phi(V)-phi(Vopt))*((xeps[:,sample]/(intepsquad*r))-gamma1/intepsquad*np.ones((N))+gamma2*weight))
	eucvec=-phip(V)/phi(V)*(phi(V)-phi(Vopt))*xeps[:,sample]
	approxvec=-gcoeff[0]*(phip(V)/phi(V)*(phi(V)-phi(Vopt))*((xeps[:,sample]/(intepsquad*r))-approxgamma1/intepsquad*np.ones((N))+approxgamma2*weight))


	norm_nat=np.sqrt(np.dot(natvec,np.dot(G,natvec)))        
	norm_euc=np.sqrt(np.dot(eucvec,np.dot(G,eucvec)))
	norm_approx=np.sqrt(np.dot(approxvec,np.dot(G,approxvec)))

	angle[sample]=np.dot(natvec,np.dot(G,eucvec))/(norm_nat*norm_euc)
	approx_angle[sample]=np.dot(natvec,np.dot(G,approxvec))/(norm_nat*norm_approx)
	
	nf[sample]=norm_euc/norm_nat
	nf_approx[sample]=norm_approx/norm_nat


angle=np.arccos(angle)
approx_angle=np.arccos(approx_angle)

mean_angle=np.mean(np.rad2deg(angle))
np.save("2019-02-12_mean_fisher_angle"+str(t)+str(N)+str(rate1)+str(rate2),mean_angle)             

mean_approx_angle=np.mean(np.rad2deg(approx_angle))
np.save("2019-02-12_mean_fisher_angle_approx"+str(t)+str(N)+str(rate1)+str(rate2),mean_approx_angle)             
       
        
np.save("2019-02-18_en_fisher_norm_fraction"+str(t)+str(N)+str(rate1)+str(rate2),np.mean(nf))        
np.save("2019-02-18_an_fisher_norm_fraction"+str(t)+str(N)+str(rate1)+str(rate2),np.mean(nf_approx))        
        
        
        
        
     
        
       
       
