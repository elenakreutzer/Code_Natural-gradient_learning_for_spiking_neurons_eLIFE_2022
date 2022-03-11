# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 14:31:43 2022

@author: Elena
"""

import File01_parameter_vectorplot as parameter
import numpy as np
import matplotlib.pyplot as plt
    
para=parameter.parameter
para["color"]="darkblue"
N=2
ax=plt.axes()
ax.set_xlim([-.3,.5])
ax.set_ylim([-.3,.5])

#################################################################################################################
def wpathplot(ax,vectorfield,weightpath,**para):
    
    target=para.get("target")
    color=para.get("color")
    wmin=para.get("wmin")
    wmax=para.get("wmax")
    precision=para.get("precision")
    #full grid
    #W1,W2=np.meshgrid(np.arange(wmin,wmax,*precision),np.arange(wmin,wmax,precision))
    #grid for sliced array
    W1,W2=np.meshgrid(np.arange(wmin,wmax,precision),np.arange(wmin,wmax,precision))
    W1_sliced= W1[2:np.shape(W1)[0]:3,3:np.shape(W1)[1]:4]
    W2_sliced= W2[2:np.shape(W2)[0]:3,3:np.shape(W2)[1]:4]
    np.save("grid_W1",W1_sliced)
    np.save("grid_W2",W2_sliced)
    ax.scatter(target[0],target[1],color=color)
    #ax.set_aspect("equal")
    ax.quiver(W2_sliced,W1_sliced,vectorfield[:,:,0],vectorfield[:,:,1],angles="xy",color=color,width=0.007,scale=10.,pivot='tail') 
    ax.plot(weightpath[0,:],weightpath[1,:],color=color, linewidth=2.)

#################################################################################################################
def contourplot(ax,cost,levels,**para):
    
    wmin=para.get("wmin")
    wmax=para.get("wmax")
    precisioncost=para.get("precisioncost")
    W1,W2=np.meshgrid(np.arange(wmin,wmax,precisioncost),np.arange(wmin,wmax,precisioncost))
    np.save("grid_W1_cost",W1)
    np.save("grid_W2_cost",W2)
    img=ax.contour(W2,W1,cost,levels=levels,marker='s',alpha=0.9,linewidths=0.5,colors="darkblue",angles="xy")
################################################################################################################
data_quiver = np.load('2019-01-07_vectorplot_euc.npy')
data_wpath=np.load("2018-08-24_ewpath.npy")
data_wpath=N*data_wpath

norm_data=np.sqrt(np.square(data_quiver[:,:,0])+np.square(data_quiver[:,:,1]))
data_quiver[:,:,0]=data_quiver[:,:,0]/norm_data
data_quiver[:,:,1]=data_quiver[:,:,1]/norm_data
data_quiver_sliced=data_quiver[2:np.shape(data_quiver)[0]:3,3:np.shape(data_quiver)[1]:4,:]

data_cost=np.load('2019-01-07_contour_cost.npy')

wpathplot(ax,data_quiver_sliced,data_wpath,**para)
np.save("vectorplot_euc",data_quiver_sliced)
#np.save("wpath_euc",data_wpath)


levels=[0.001,0.003,0.005,0.009,0.015,0.02,0.03,0.04]
contourplot(ax,data_cost, levels=levels,**para)
#np.save("cost_contour",data_cost)
#normalize

ax.get_xaxis().set_visible(True)
ax.get_yaxis().set_visible(True)

ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)


ax.set_xlabel(r'$w_\mathrm{1}$ [a.u.]')
ax.set_ylabel(r'$w_\mathrm{2}$ [a.u.]')

###################################################################################################################
  
para=parameter.parameter
para["color"]="indianred"



data_quiver = np.load('2019-01-07_vectorplot_nat.npy')

data_cost=np.load('2019-01-07_contour_cost.npy')
data_wpath=np.load("2018-08-24_nwpath.npy")
data_wpath=N*data_wpath

#normalize
norm_data=np.sqrt(np.square(data_quiver[:,:,0])+np.square(data_quiver[:,:,1]))
data_quiver[:,:,0]=data_quiver[:,:,0]/norm_data
data_quiver[:,:,1]=data_quiver[:,:,1]/norm_data
data_quiver_sliced=data_quiver[2:np.shape(data_quiver)[0]:3,3:np.shape(data_quiver)[1]:4,:]
wpathplot(ax,data_quiver_sliced,data_wpath,**para)
np.save("vectorplot_nat",data_quiver_sliced)
np.save("wpath_nat",data_wpath)

levels=[0.001,0.003,0.005,0.009,0.015,0.02,0.03,0.04]
contourplot(ax,data_cost, levels=levels,**para)
ax=plt.axes()
   

ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.get_xaxis().set_visible(True)
ax.get_yaxis().set_visible(True)


ax.set_xlabel(r'$w_\mathrm{1}$ [a.u.]')
ax.set_ylabel(r'$w_\mathrm{2}$ [a.u.]')
plt.show()
