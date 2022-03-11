#!/usr/bin/env python2
# encoding: utf-8

import matplotlib as mpl
reload(mpl)


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
import numpy.ma as ma
from mpl_toolkits.axes_grid1 import make_axes_locatable







###############################
# Plot functions for subplots #
###############################
#
# naming scheme: plot_<key>(ax)
#
# ax is the Axes to plot into
    
#

class MidpointNormalize(Normalize):
	def __init__(self,vmin=None,vmax=None,midpoint=None,clip=True):
		self.midpoint=midpoint
		Normalize.__init__(self,vmin,vmax,clip)
	def __call__(self,value,clip=True):
		x,y=[self.vmin,self.midpoint,self.vmax],[0.,0.5,1.]
		return(np.ma.masked_array(np.interp(value,x,y)))

#adjust colormap
cmap = mpl.cm.RdBu(np.linspace(0,1,100))
cmap = mpl.colors.ListedColormap(cmap[np.concatenate((np.arange(0,30,1),np.arange(70,100,1)))])
def hplot(ax,data,vmin,vmax,extent,label,labelpad,cmap):
    norm=MidpointNormalize(vmin=vmin,vmax=vmax,midpoint=0)
    #norm=mpl.colors.SymLogNorm(linthresh=0.00003, linscale=0.00003,vmin=vmin, vmax=vmax)
   # zeros=np.absolute(data)<10**(-3)
    #data[zeros]=0
    
    
    #,norm=norm
    img=ax.imshow(data,cmap=cmap,norm=norm,origin="lower",interpolation="none",extent=extent.tolist(),zorder=1)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad=0.02)


    cb= plt.colorbar(img,cax=cax)
     
  
    cbar_ticks = np.array((int(vmin),0,int(vmax)))
    cb.set_ticks(cbar_ticks) 
    cb.ax.tick_params(labelsize=6)
    cb.draw_all() 
    cb.outline.set_visible(False)
    cax.set_aspect(12)
    cb.update_ticks()    





def plot_heatplot_homsyn(ax):
    ax.set_xlim([1.,5.])
    ax.set_ylim([1.,5.])
    data = np.load('Delta_w_hom_grid_1min.npy')
    data=data[100:,100:]
    np.save("data_heatplot_hom",data)
    data_aux = np.load('Delta_w_het_grid_1min.npy')
    data_aux=data_aux[100:,100:]
    w_range = np.load('w_range.npy')
    initialweight=0.01*np.linspace(0.25,5.,600.)[100:]
    
    data=np.divide(data,initialweight[:,np.newaxis])
    data_aux=np.divide(data_aux,initialweight[np.newaxis,:])
    
    ax.scatter(x=[1.6,2.5,4.5],y=[1.6,2.5,4.5],zorder=2, s=2.,color='black')
    
    x_ticks = np.array((2.,4.,))
    ax.xaxis.set_ticks(x_ticks)
    y_ticks = np.array((2.,4.,))
    ax.yaxis.set_ticks(y_ticks) 
    wmin=np.linspace(0.25,5.,600.)[100]
    w_range=np.array([wmin,5.,wmin,5.])
    
    
    vmin=-12.
    vmax=12.
    
    
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    
    label=r"$\Delta w_{\rm{st}}$ [$\%$]"
    labelpad=-8.
    #ax.set_title(label,fontsize=core.linewidth["cb"],position=(.6,1.))
    
    cmap = mpl.cm.seismic(np.linspace(0,1,500)[::-1])
    cmap = mpl.colors.ListedColormap(cmap[np.concatenate((np.arange(0,242,1),np.arange(258,500,1)))])
        
    hplot(ax,data,vmin,vmax,w_range,label,labelpad,cmap=cmap)
    np.save("data_heatplot_hom",data)
    np.save("weightrange",w_range)
   
    ax.set_xlabel(r'$w_{\mathrm{ust}}(t_0)$ [a.u.]')
    ax.set_ylabel(r'$w_{\mathrm{st}}(t_0)$ [a.u.]')


    pass

def plot_heatplot_heterosyn(ax):
    ax.set_xlim([1.,5.])
    ax.set_ylim([1.,5.])
   
    data =np.load('Delta_w_het_grid_1min.npy')
    
    data=data[100:,100:]
    data_aux=np.load('Delta_w_hom_grid_1min.npy')
    data_aux=data_aux[100:,100:]
    initialweight=0.01*np.linspace(0.25,5.,600.)[100:]
    
    
    data=np.divide(data,initialweight[np.newaxis,:])
    data_aux=np.divide(data_aux,initialweight[:,np.newaxis])
    
    ax.scatter(x=[1.6,2.5,4.5],y=[1.6,2.5,4.5],zorder=2, s=2.,color='black')
    
    x_ticks = np.array((2.,4.,))
    ax.xaxis.set_ticks(x_ticks)
    y_ticks = np.array((2.,4.,))
    ax.yaxis.set_ticks(y_ticks) 
    
    vmin=min(np.amin(data),np.amin(data_aux))
    vmax=max(np.amax(data),np.amax(data_aux))
    
    vmin=-12.
    vmax=12.
    
    w_range = np.load('w_range.npy')
    wmin=np.linspace(0.25,5.,600.)[100]
    w_range=np.array([wmin,5.,wmin,5.])

    
    
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    
    label=r"$\Delta w_{\rm{ust}}$ [$\%$]"
  
    labelpad=-8.
    
    cmap = mpl.cm.seismic(np.linspace(0,1,500)[::-1])
    cmap = mpl.colors.ListedColormap(cmap[np.concatenate((np.arange(0,242,1),np.arange(258,500,1)))])

    hplot(ax,data,vmin,vmax,w_range,label,labelpad,cmap=cmap)
    np.save("data_heatplot_het",data)
        
    ax.set_xlabel(r'$w_{\mathrm{ust}}(t_0)$ [a.u.]')
    ax.set_ylabel(r'$w_{\mathrm{st}}(t_0)$ [a.u.]')
   

    pass
def plot_heatplot_hom_het_comparison(ax):
    

    data = np.load('Sign_delta_w_comparison_1min.npy')
    data=data[100:,100:]
    
    x_ticks = np.array((0,2.,4.,))
    ax.xaxis.set_ticks(x_ticks)
    y_ticks = np.array((0,2.,4.,))
    ax.yaxis.set_ticks(y_ticks)    
    
    W1,W2=np.meshgrid(np.linspace(0.,5.,10.),np.linspace(0.,5.,10.))
    ax.scatter(x=[1.6,2.5],y=[1.6,2.5],zorder=2, s=2.,color='black')
    ax.scatter(x=[4.5],y=[4.5],zorder=3, s=2.,color='white')
    w_range = np.load('w_range.npy')
    
    wmin=np.linspace(0.25,5.,600.)[100]
    w_range=np.array([wmin,5.,wmin,5.])

    

    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    
       

    cmap = mpl.cm.PuBuGn(np.linspace(0,1,100))
    
    cmap = mpl.colors.ListedColormap(cmap[np.concatenate((np.arange(80,100,1),np.arange(0,20,1)))])
    cmap.set_bad('lightseagreen',0.1)
    masked_data=ma.masked_where(data<0,data)
    

    img=ax.imshow(masked_data,cmap=cmap,origin="lower",extent=w_range.tolist(),zorder=2)
    
    ax.set_xlabel(r'$w_{\mathrm{ust}}(t_0)$ [a.u.]')
    ax.set_ylabel(r'$w_{\mathrm{st}}(t_0)$ [a.u.]')
   
    
    
    pass


plt.figure(1)
    
ax=plt.axes()
plot_heatplot_homsyn(ax)
plt.show()


plt.figure(2)
    
ax=plt.axes()
plot_heatplot_heterosyn(ax)
plt.show()

plt.figure(3)
    
ax=plt.axes()
plot_heatplot_hom_het_comparison(ax)
plt.show()   
   




