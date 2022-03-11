# -* coding: utf-8 -*-
"""
Created on Mon May 23 11:09:39 2016

@author: kreutzer
"""
#==============================================================================
# Import modules
#==============================================================================
import os
import numpy as np
import scipy as sp
import scipy.integrate as spig
import File01_parameter_angle as parameter

t=int(os.environ['SLURM_ARRAY_TASK_ID'])

#==============================================================================
# Set the parameters
#==============================================================================

para=parameter.parameter



#number of neurons
N=para.get("N")



#time constants of the PSP kernel
taul=para.get("taul")
taus=para.get("taus")
intepsquad=para.get("intepsquad")


#scale for nonlinearity
maxfire=para.get("maxfire")
threshold=para.get("threshold")
slope=para.get("slope")



#============================================================================
#define transfer function
#===========================================================================

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



	
def g(m,v,q):
        I1=(slope**2)*(1./(np.sqrt(2.*np.pi*v)))*spig.quad(lambda u: maxfire*phi(u)*((1.-phi(u))**2)*np.exp(-0.5*((u-m)**2)/v),-np.inf,np.inf)[0]
        I2=(slope**2)*(1./(np.sqrt(2.*np.pi*v)))*spig.quad(lambda u: maxfire*phi(u)*((1.-phi(u))**2)*u*np.exp(-0.5*((u-m)**2)/v),-np.inf,np.inf)[0]
        I3=(slope**2)*(1./(np.sqrt(2.*np.pi*v)))*spig.quad(lambda u: maxfire*phi(u)*((1.-phi(u))**2)*(u**2)*np.exp(-0.5*((u-m)**2)/v),-np.inf,np.inf)[0]
                  

        c2=(I2-I1*m)/(v)
        c3=(I3-I1*((m**2)+v)-2.*c2*m*v)/(v**2)
  


        return(np.array([I1,c2,c3]))








#=========================================================================================
#lookup table
#=========================================================================================
#q is pa.rallelized, for two to hundred  neurons and rates between 10 and 50, q between 0.5 and 100 makes sense
q=t
lt=np.zeros((122,820,3))
#given a sigmoidal center around 60, and a saturation of -70, width is approximately 10, so choosing m between -15, this yields  wmin=-0.5, wmax=0.5, and with intepsquad approximataly 40, the range for v follows

for i,mi in enumerate(np.arange(-15.,15.5,0.25)):
	for j,vj in enumerate(np.arange(10.,411.,.5)):
		lt[i,j,:]=g(mi,vj,q)
	
np.save("2019-02-13_lt_c"+str(t),lt)

