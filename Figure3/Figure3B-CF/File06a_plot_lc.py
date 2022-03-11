# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 13:19:25 2022

@author: Elena
"""

import numpy as np
import matplotlib.pyplot as plt

data_nat=np.load("2018-08-24_naturalgradient_KL_error.npy")
data_euc=np.load("2018-08-24_euclideangradient_KL_error.npy")
data_nat_min=np.load("2018-08-24_naturalgradient_KL_error_min.npy")
data_euc_min=np.load("2018-08-24_euclideangradient_KL_error_min.npy")
data_nat_max=np.load("2018-08-24_naturalgradient_KL_error_max.npy")
data_euc_max=np.load("2018-08-24_euclideangradient_KL_error_max.npy")
#data_approx=np.load("2018-08-24_approxgradient_KL_error.npy")
data_time=np.arange(0.,81.,1.)*100./60.


fig=plt.figure(0)
ax=plt.axes()
ax.get_xaxis().set_visible(True)
ax.get_yaxis().set_visible(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.set_yscale('log')

ax.semilogy(data_time,data_euc,color="darkblue",linewidth=2.)
ax.semilogy(data_time,data_nat,color="indianred",linewidth=2.)
#ax.semilogy(data_time,data_approx,color="orange",linewidth=2.)

fig=plt.figure(1)
ax=plt.axes()
ax.get_xaxis().set_visible(True)
ax.get_yaxis().set_visible(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.set_yscale('log')

ax.semilogy(data_time,data_euc,color="darkblue",linewidth=2.)
ax.semilogy(data_time,data_nat,color="indianred",linewidth=2.)

ax.semilogy(data_time,data_euc_min,color="darkblue",linewidth=2.,alpha=0.3)
ax.semilogy(data_time,data_nat_min,color="indianred",linewidth=2.,alpha=0.3)

ax.semilogy(data_time,data_euc_max,color="darkblue",linewidth=2.,alpha=0.3)
ax.semilogy(data_time,data_nat_max,color="indianred",linewidth=2.,alpha=0.3)

ax.set_xlabel(r'$t$ [min]')
ax.set_ylabel(r"$D_{\mathrm{KL}}(p^* | p_{w})$")
ax.legend(("Euclidean gradient","natural gradient"),loc=0,fontsize=8.,labelspacing=0.1,borderaxespad=0.1,borderpad=0.,handlelength=0.5,frameon=False)

#####################################################################################################################

data_nat=np.load("2021-08-21_n_wdist_mean.npy")
data_euc=np.load("2021-08-21_e_wdist_mean.npy")
data_time=np.arange(0.,81.,1.)*100/60.
fig2=plt.figure(2)
ax=plt.axes()
ax.get_xaxis().set_visible(True)
ax.get_yaxis().set_visible(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

ax.plot(data_time,data_euc,color="darkblue",linewidth=2.)
ax.plot(data_time,data_nat,color="indianred",linewidth=2.)


ax.set_xlabel(r'$t$ [min]')
ax.set_ylabel(r"$\|w-w^*\|_{\rm e}$")
ax.legend(("Euclidean gradient","natural gradient"),loc=0,fontsize=8.,labelspacing=0.1,borderaxespad=0.1,borderpad=0.,handlelength=0.5,frameon=False)
#####################################################################################################################


data_nat=np.load("2018-08-24_naturalgradient_rate_error.npy")
data_euc=np.load("2018-08-24_euclideangradient_rate_error.npy")
data_time=np.arange(0.,81.,1.)*100./60.
#
fig3=plt.figure(3)
ax=plt.axes()
ax.get_xaxis().set_visible(True)
ax.get_yaxis().set_visible(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
    
ax.semilogy(data_time,data_euc,color="darkblue",linewidth=2.)
ax.semilogy(data_time,data_nat,color="indianred",linewidth=2.)
     
    
ax.set_xlabel(r'$t$ [min]')
ax.set_ylabel(r"$\|\phi-\phi^*\|_e$")
ax.legend(("Euclidean gradient","natural gradient"),loc=0,fontsize=8.,labelspacing=0.1,borderaxespad=0.1,borderpad=0.,handlelength=0.5,frameon=False)