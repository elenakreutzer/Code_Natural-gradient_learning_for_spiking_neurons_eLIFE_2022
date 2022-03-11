# -* coding: utf-8 -*-
"""
Created on Mon May 23 11:09:39 2016

@author: kreutzer
"""
#==============================================================================
# Import modules
#==============================================================================
import os                                                                      #module needed to run multiple instances on SLURM cluster
import numpy as np                                                             #import numpy module
import scipy as sp
import scipy.integrate as spig                                                 #scipy module neede for integration
import File01_simulation_parameters as parameter                               #import file that contains all simulation parameters

t=int(os.environ['SLURM_ARRAY_TASK_ID'])                                       #run multiple instances on a slurm cluster, disable if only a single instance should be run manually

#==============================================================================
# Set the parameters
#==============================================================================

para=parameter.parameter                                                       #see comments in "File01_simulation_parameters" for explanations of the different parameters



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

#sigmoidal transfer function
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



	
def g(m,v,q):                                                                  #calculate the coefficients g_1, ...g_4 and gamma_s
        
        # Eqn. 77-79
        I1=(slope**2)*(1./(np.sqrt(2.*np.pi*v)))*spig.quad(lambda u: maxfire*phi(u)*((1.-phi(u))**2)*np.exp(-0.5*((u-m)**2)/v),-np.inf,np.inf)[0]
        I2=(slope**2)*(1./(np.sqrt(2.*np.pi*v)))*spig.quad(lambda u: maxfire*phi(u)*((1.-phi(u))**2)*u*np.exp(-0.5*((u-m)**2)/v),-np.inf,np.inf)[0]
        I3=(slope**2)*(1./(np.sqrt(2.*np.pi*v)))*spig.quad(lambda u: maxfire*phi(u)*((1.-phi(u))**2)*(u**2)*np.exp(-0.5*((u-m)**2)/v),-np.inf,np.inf)[0]
                  
        #Eqn. 82
        c2=(I2-I1*m)/(v)
        c3=(I3-I1*((m**2)+v)-2.*c2*m*v)/(v**2)
        
        #Eqn. 88, K1=k1*c1
        K1=1./(I1*(q+1)+c2*m)
        #Eqn. 91, K2=k2/c1
        K2=1./(I1+(c2*m+c3*v)-K1*(I1*m+c2*v)*(c2*q+c3*m))
  
        #Eqn. 102-105
        g1=K1*K2*(-1./K2*I1+(I1*c2*m+(c2**2)*v)-(K1)*I1*((I1*c2*m+v*c2**2)*q+(I1*c3*m+c2*c3*v)*m))

        g2=K1*K2*(-c2/K2+(I1*c3*m+c2*c3*v)*(1.-K1*m*c2)-K1*c2*(I1*c2*m+v*c2**2)*q)

        g3=K1*K2*((I1*c2*q+I1*c3*m)-c2/K1)

        g4=K1*K2*(((c2**2)*q+c2*c3*m)-c3/K1)

        return(np.array([1./I1,g1,g2,g3,g4]))








#=========================================================================================
#lookup table
#=========================================================================================
#for every q, a single instance is run on the cluster. The necessary values for q are completely determined by the afferent input rates and time constants (see Eqn. 36).
q=t
lookup_table=np.zeros((122,820,5))
#given a sigmoidal center around 60, and a saturation of -70, we choose a range for mean V between -15 and 15 and the range for the variance of V between 10. and 410.

for i,meanV_i in enumerate(np.arange(-15.,15.5,0.25)):
	for j,varianceV_j in enumerate(np.arange(10.,411.,.5)):
		lookup_table[i,j,:]=g(meanV_i,varianceV_j,q)
	
np.save("2019-01-06_lt_newtf"+str(t),lookup_table)

