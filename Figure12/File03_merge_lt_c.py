import numpy as np
lt=np.zeros((13,122,820,3))
for counter,value in enumerate((5,7,26,39,52,78,260,390,520,780,2600,3900,7800)):
	lt[counter,:,:,:]=np.load("2019-02-13_lt_c"+str(value)+".npy")
np.save("2019-02-13_lookup_table_c.npy",lt)
