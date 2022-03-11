# -* coding: utf-8 -*-
"""
Created on Mon May 23 11:09:39 2016

@author: kreutzer
"""
#==============================================================================
# Import modules
#==============================================================================
import numpy as np
import scipy.integrate as spig
import File01_parameter_vectorplot as parameter


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
def rho(u):
    if u>=0.:
        return(1./(1.+np.exp(-slope*(u-threshold))))#threshold=center sigmoidal -resting_potential
    else:
        return(np.exp(slope*(u-threshold))/(1.+np.exp(slope*(u-threshold))))


#vectorize nonlinearity
vrho=np.vectorize(rho)


#derivative of nonlinearity
def rhop(u):
      return(slope*rho(u)*(1.-rho(u)))



	
def f(m,v,q):
        I1=(slope**2)*(1./(np.sqrt(2.*np.pi*v)))*spig.quad(lambda u: maxfire*rho(u)*((1.-rho(u))**2)*np.exp(-0.5*((u-m)**2)/v),-np.inf,np.inf)[0]
        I2=(slope**2)*(1./(np.sqrt(2.*np.pi*v)))*spig.quad(lambda u: maxfire*rho(u)*((1.-rho(u))**2)*u*np.exp(-0.5*((u-m)**2)/v),-np.inf,np.inf)[0]
        I3=(slope**2)*(1./(np.sqrt(2.*np.pi*v)))*spig.quad(lambda u: maxfire*rho(u)*((1.-rho(u))**2)*(u**2)*np.exp(-0.5*((u-m)**2)/v),-np.inf,np.inf)[0]
                  

        c2=(I2-I1*m)/(v)
        c3=(I3-I1*((m**2)+v)-2.*c2*m*v)/(v**2)
        K1=1./(I1*(q+1)+c2*m)
        K2=1./(I1+(c2*m+c3*v)-K1*(I1*m+c2*v)*(c2*q+c3*m))
  

        g1=K1*K2*(-1./K2*I1+(I1*c2*m+(c2**2)*v)-(K1)*I1*((I1*c2*m+v*c2**2)*q+(I1*c3*m+c2*c3*v)*m))

        g2=K1*K2*(-c2/K2+(I1*c3*m+c2*c3*v)*(1.-K1*m*c2)-K1*c2*(I1*c2*m+v*c2**2)*q)

        g3=K1*K2*((I1*c2*q+I1*c3*m)-c2/K1)

        g4=K1*K2*(((c2**2)*q+c2*c3*m)-c3/K1)

        return(np.array([1./I1,g1,g2,g3,g4]))








#=========================================================================================
#lookup table
#=========================================================================================

q=1.5
lt=np.zeros((122,820,5))
#given a sigmoidal center around 60, and a saturation of -70, width is approximately 10, so choosing m between -15, this yields  wmin=-0.5, wmax=0.5, and with intepsquad approximataly 40, the range for v follows

for i,mi in enumerate(np.arange(-15.,15.5,0.25)):
	for j,vj in enumerate(np.arange(10.,411.,.5)):
		lt[i,j,:]=f(mi,vj,q)
	
np.save("2019-01-06_lt_newtf"+str(q),lt)

