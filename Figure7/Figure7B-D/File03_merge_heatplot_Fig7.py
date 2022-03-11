#########################################################################
#import modules
#########################################################################
import numpy as np
from File01_weightrange_Fig7 import wr



w_change_hom,w_change_het=np.zeros((2,wr["steps"],wr["steps"]))
w_range=np.array((wr["w_min"],wr["w_max"],wr["w_min"],wr["w_max"]))
np.save("w_range",w_range)

for i in range(wr["steps"]):
##	startweight=np.load("hom"+str(i+1)+".npy")
	w_change_hom[i,:]=np.load("Delta_w_hom"+str(i+1)+".npy")
	w_change_het[i,:]=np.load("Delta_w_het"+str(i+1)+".npy")
np.save("Delta_w_hom_grid_1min",w_change_hom)
np.save("Delta_w_het_grid_1min",w_change_het)


sign_hom=np.sign(w_change_hom)
np.save("Sign_delta_w_hom_1min",sign_hom)

sign_het=np.sign(w_change_het)
np.save("Sign_delta_w_het_1min",sign_het)
sign_comparison=np.sign(np.multiply(w_change_hom,w_change_het))
np.save("Sign_delta_w_comparison_1min",sign_comparison)
