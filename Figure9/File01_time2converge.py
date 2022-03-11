# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 13:00:24 2022

@author: Elena
"""


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def plot_learningrate_tuning(ax):
    
    data_nat_200=np.load("2018-08-24_naturalgradient_KL_error_200.npy")
    data_euc_200=np.load("2018-08-24_euclideangradient_KL_error_200.npy")
    data_nat_180=np.load("2018-08-24_naturalgradient_KL_error_180.npy")
    data_euc_180=np.load("2018-08-24_euclideangradient_KL_error_180.npy")
    data_nat_160=np.load("2018-08-24_naturalgradient_KL_error_160.npy")
    data_euc_160=np.load("2018-08-24_euclideangradient_KL_error_160.npy")
    data_nat_140=np.load("2018-08-24_naturalgradient_KL_error_140.npy")
    data_euc_140=np.load("2018-08-24_euclideangradient_KL_error_140.npy")
    data_nat_120=np.load("2018-08-24_naturalgradient_KL_error_120.npy")
    data_euc_120=np.load("2018-08-24_euclideangradient_KL_error_120.npy")
    data_nat_100=np.load("2018-08-24_naturalgradient_KL_error_100.npy")
    data_euc_100=np.load("2018-08-24_euclideangradient_KL_error_100.npy")
    data_nat_90=np.load("2018-08-24_naturalgradient_KL_error_90.npy")
    data_euc_90=np.load("2018-08-24_euclideangradient_KL_error_90.npy")
    data_nat_80=np.load("2018-08-24_naturalgradient_KL_error_80.npy")
    data_euc_80=np.load("2018-08-24_euclideangradient_KL_error_80.npy")
    data_nat_70=np.load("2018-08-24_naturalgradient_KL_error_70.npy")
    data_euc_70=np.load("2018-08-24_euclideangradient_KL_error_70.npy")
    data_nat_60=np.load("2018-08-24_naturalgradient_KL_error_60.npy")
    data_euc_60=np.load("2018-08-24_euclideangradient_KL_error_60.npy")
    data_nat_50=np.load("2018-08-24_naturalgradient_KL_error_50.npy")
    data_euc_50=np.load("2018-08-24_euclideangradient_KL_error_50.npy")
    
    

    cutoff=0.00005
       
    index_200_nat=np.min(np.where(np.array(data_nat_200)<cutoff))
    index_180_nat=np.min(np.where(np.array(data_nat_180)<cutoff))
    index_160_nat=np.min(np.where(np.array(data_nat_160)<cutoff))
    index_140_nat=np.min(np.where(np.array(data_nat_140)<cutoff))
    index_120_nat=np.min(np.where(np.array(data_nat_120)<cutoff))
    index_100_nat=np.min(np.where(np.array(data_nat_100)<cutoff))
    index_90_nat=np.min(np.where(np.array(data_nat_90)<cutoff))
    index_80_nat=np.min(np.where(np.array(data_nat_80)<cutoff))
    index_70_nat=np.min(np.where(np.array(data_nat_70)<cutoff))
    index_60_nat=np.min(np.where(np.array(data_nat_60)<cutoff))
    index_50_nat=np.min(np.where(np.array(data_nat_50)<cutoff))
    ax.scatter(np.array(((2,1.8,1.6,1.4,1.2,1,.9,.8,.7,.6,.5))), np.array((index_200_nat,index_180_nat,index_160_nat,index_140_nat,index_120_nat,index_100_nat,index_90_nat,index_80_nat,index_70_nat,index_60_nat,index_50_nat))*100./60., marker='o',color="indianred",s=15.)
    
    
    index_200_euc=np.min(np.where(np.array(data_euc_200)<cutoff))
    index_180_euc=np.min(np.where(np.array(data_euc_180)<cutoff))
    index_160_euc=np.min(np.where(np.array(data_euc_160)<cutoff))
    index_140_euc=np.min(np.where(np.array(data_euc_140)<cutoff))
    index_120_euc=np.min(np.where(np.array(data_euc_120)<cutoff))
    index_100_euc=np.min(np.where(np.array(data_euc_100)<cutoff))
    index_90_euc=np.min(np.where(np.array(data_euc_90)<cutoff))
    index_80_euc=np.min(np.where(np.array(data_euc_80)<cutoff))
    index_70_euc=np.min(np.where(np.array(data_euc_70)<cutoff))
    index_60_euc=np.min(np.where(np.array(data_euc_60)<cutoff))
    index_50_euc=np.min(np.where(np.array(data_euc_50)<cutoff))
    ax.scatter(np.array(((2,1.8,1.6,1.4,1.2,1,.9,.8,.7,.6,.5))),np.array((index_200_euc,index_180_euc,index_160_euc,index_140_euc,index_120_euc,index_100_euc,index_90_euc,index_80_euc,index_70_euc,index_60_euc, index_50_euc))*100./60., marker='o',color="darkblue",s=15.)
    

    
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    ax.set_xlim(.48,2.2)
    ax.set_xlabel(r"$\eta/\eta_0$")
    ax.set_ylabel("Conv. time [min]")
    
    
plt.figure(1)
ax=plt.axes()
plot_learningrate_tuning(ax)