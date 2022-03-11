import numpy as np
import File01_parameter_angle as parameter
para=parameter.parameter
N=para["N"]
samplenumber=para["samplenumber"]
wmin=para["wmin"]
wmax=para["wmax"]
weightsamples=para["weightsamples"]
rate1=para["rate1"]
rate2=para["rate2"]
angle,approx_angle,nf,nf_approx=np.zeros((4,weightsamples))
for index in range(weightsamples):
	angle[index]=np.load("2019-02-12_mean_fisher_angle"+str(index+1)+str(N)+str(rate1)+str(rate2)+".npy")
	approx_angle[index]=np.load("2019-02-12_mean_fisher_angle_approx"+str(index+1)+str(N)+str(rate1)+str(rate2)+".npy")
	
	
np.save("2019-02-18_en_fisher_angle"+str(N)+str(rate1)+str(rate2),angle)
np.save("2019-02-18_an_fisher_angle"+str(N)+str(rate1)+str(rate2),approx_angle)

