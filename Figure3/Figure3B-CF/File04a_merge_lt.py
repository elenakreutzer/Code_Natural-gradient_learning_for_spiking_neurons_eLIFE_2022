import numpy as np
import File01_simulation_parameters  as para
lt=np.zeros((13,122,820,5))
for counter,value in enumerate((5,7,26,39,52,78,260,390,520,780,2600,3900,7800)):
	lt[counter,:,:,:]=np.load("2019-01-06_lt_newtf"+str(value)+".npy")
np.save("2019-01-06_lookup_table_big.npy",lt)
