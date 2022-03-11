# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 17:55:06 2019

@author: Elena
"""
import numpy as np

parameter={
    "samplenumber":2000,
    "threshold":10.,
    "slope":0.3,
    "maxfire":100.,
    "wmin":-0.4,
    "wmax":0.6,
    "N":2,
    "taul":0.01,
    "taus":0.003,
    "inteps":1.,
    "intepsquad":38.46,
    "dt":0.0005,
    "rate1":10.,# change lookup table if adapted
    "rate2":50.,#change lookup table if adapted
    "timesteps":2000,
    "target":np.array((0.15,0.15)),
    "precisioncost":0.006,
    "precision":0.06,#0.25,
    "lt":"2019-01-06_lt_newtf1.5.npy",
 
}
