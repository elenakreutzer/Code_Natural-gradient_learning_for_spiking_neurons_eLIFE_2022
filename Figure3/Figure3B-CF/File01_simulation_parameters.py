# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 11:44:55 2018

@author: Elena
"""

parameter={
    #number of parallel instances on the cluster
	"number_instances":1000,
 
    #learningrates
    "learningrate_natural":0.0006,              #learning rate for natural-gradient-descent algorithm (Eqn. 13)
    "learningrate_euclidean":0.00000045,        #learning rate for Euclidean-gradient-descent algorithm (Eqn. 10)
    "learningrate_approx":0.00055,              #learning rate for approximated natural gradient descent learning rule (Eqn. 114)
    #number of neurons
    "N":100,

	#weight range
	"wmin":-1.,                                 #interval from which each initial and target weight component is sampled (will be scaled by 1./N)
	"wmax":1.,
        
    #input rates
    "rate1":10.,                                #Possion firing rate of presynaptic afferents group 1
    "rate2":50.,                                #Possion firing rate of presynaptic afferents group 2
        
	                   

    #PSP kernel time constants (see Eqn. 28)
    "taul":0.01,                                #membrane time constant
    "taus":0.003,                               #synaptic time constant
    "fac":(1./0.007),                           #prefactor in Eqn. 28 , fac=(1./((taul-taus)))
    "inteps":1. ,                               #integral over the input kernel inteps=fac*(taul-taus)
    "intepsquad":38.46,                         #integral over the squared input kernel(fac**2)*(taul/2.-(2./(1./taul+1./taus))+taus/2., intepsquad=c_eps^{-1}
	
    #intervals for learning 
	"l_intervals":8000,

    #timesteps for learning (per interval, changing requires rerun of sample_input_spiketrains.sh)
    "timesteps":2000,
        
    #Euler timestep
    "dt":0.0005,
        
    #parameters of the sigmoidal transfer function (see Eqn. 24, changing these parameters requires rerun of lookup table for the coefficients g_1,...g_4)
    "slope":0.3,                                #slope=beta in Eqn. 24
    "threshold":10.,                            #threshold=theta in Eqn. 24
    "maxfire":100.,                             #maxfire=phi_max in Eqn.24
        
    #lookup table
    "lt_file":"2019-01-06_lookup_table_big.npy",#file that contains pre-calculated look-up table for the coefficients g_1,..,g_4
        
    #target weights for weight path plot (Fig. 3D&E), for the other simulations target is randomly sampled
    "target1":0.15,                             #first component of target weight vector, will be multiplied by 1./N
    "target2":0.15,                             #second component of target weight vector, will be multiplied by 1./N
        
    #initial weights for weight path plot (Fig. 3D&E), for the other simulations initial weight is randomly sampled
    "initial1":0.3,                             #first component of initial weight vector, will be multiplied by 1./N
    "initial2":0.3,                             #second component of initial weight vector, will be multiplied by 1./N
        
	#alphas for attenuation
	"alpha1":1.,                                #refers to 1/f' in Eqn. 13 and alpha in Eqn. 17. For Figures3,8,and 12 \\alpha=1., since simulations are done in somatic paramterization.
	"alpha2":1.,

    #interval for error calculation
    "interval":100,                             #avoid data set becoming to large by not calculating error every timestep.
    
    #spike train samples (must coincide with number in sample_inputspikes.sh)
	"sample_spiketrains":8000,                  #number of pregenerated spiketrains of length timesteps
        
    #timesteps and samplenumber for test spike trains
    "testtimesteps":500,                        #randomly generate test set to evaluate mean suqared error in rate and DKL
    "testsample":50,
        
    "index":10,                                 #index for psth example
	}
