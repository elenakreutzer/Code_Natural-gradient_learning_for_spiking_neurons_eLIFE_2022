# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 17:55:52 2022

@author: Elena
"""

import numpy as np
from cycler import cycler
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_fisher_angles(ax):
    
    data1 = np.load('2019-02-18_an_fisher_angle10010.010.0.npy')
    data2 = np.load('2019-02-18_an_fisher_angle10020.020.0.npy')
    data3 = np.load('2019-02-18_an_fisher_angle10010.030.0.npy')
    data4 = np.load('2019-02-18_an_fisher_angle10020.040.0.npy')
    data5 = np.load('2019-02-18_an_fisher_angle10010.050.0.npy')
    
    np.save('approx_nat_fisher_angle10010.010',data1)
    np.save('approx_nat_fisher_angle10020.020',data2)
    np.save('approx_nat_fisher_angle10010.030',data3)
    np.save('approx_nat_fisher_angle10020.040',data4)
    np.save('approx_nat_fisher_angle10010.050',data5)
    
    
    data1euc = np.load('2019-02-18_en_fisher_angle10010.010.0.npy')
    data2euc = np.load('2019-02-18_en_fisher_angle10020.020.0.npy')
    data3euc = np.load('2019-02-18_en_fisher_angle10010.030.0.npy')
    data4euc = np.load('2019-02-18_en_fisher_angle10020.040.0.npy')
    data5euc = np.load('2019-02-18_en_fisher_angle10010.050.0.npy')
    
    np.save('euc_nat_fisher_angle10010.010',data1euc)
    np.save('euc_nat_fisher_angle10020.020',data2euc)
    np.save('euc_nat_fisher_angle10010.030',data3euc)
    np.save('euc_nat_fisher_angle10020.040',data4euc)
    np.save('euc_nat_fisher_angle10010.050',data5euc)
    
    data=np.concatenate((data1,data2,data3,data4,data5))
    dataeuc=np.concatenate((data1euc,data2euc,data3euc,data4euc,data5euc))

    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    
    ax.set_prop_cycle(cycler('color',['orange','darkblue'] ))
    
    bins=np.linspace(0.,90.,18.,endpoint=False)
    trialnumber=float(len(data))
    print(trialnumber)
    trialnumber_euc=float(len(dataeuc))
    print(trialnumber_euc)
    ax.hist([data,dataeuc], bins,normed=0.)
    xmin=0.
    xmax=90.
    ymin=0.
    ymax=250.
    deltax=xmax-xmin
    deltay=ymax-ymin
    xrange=np.array([xmin,xmax]) 
    yrange=np.array([ymin,ymax])
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)
    x_ticks = np.array((xmin+0.*deltax,xmin+0.5*deltax,xmin+1.0*deltax))
    y_ticks = np.array((ymin+0.*deltay,ymin+0.5*deltay,ymin+1.0*deltay))
    ax.set_yticklabels(y_ticks/trialnumber)
    ax.xaxis.set_ticks(x_ticks)
    ax.yaxis.set_ticks(y_ticks)
    
    ax.set_xlabel("angle Euclidean metric [deg]")
    ax.set_ylabel("fraction of samples")
    
    pass


def plot_euc_angles(ax):
    
    data1 = np.load('2019-02-18_an_euc_angle10010.010.0.npy')
    data2 = np.load('2019-02-18_an_euc_angle10020.020.0.npy')
    data3 = np.load('2019-02-18_an_euc_angle10010.030.0.npy')
    data4 = np.load('2019-02-18_an_euc_angle10020.040.0.npy')
    data5 = np.load('2019-02-18_an_euc_angle10010.050.0.npy')
    
    np.save('approx_nat_euc_angle10010.010',data1)
    np.save('approx_nat_euc_angle10020.020',data2)
    np.save('approx_nat_euc_angle10010.030',data3)
    np.save('approx_nat_euc_angle10020.040',data4)
    np.save('approx_nat_euc_angle10010.050',data5)
    
    data1euc = np.load('2019-02-18_en_euc_angle10010.010.0.npy')
    data2euc = np.load('2019-02-18_en_euc_angle10020.020.0.npy')
    data3euc = np.load('2019-02-18_en_euc_angle10010.030.0.npy')
    data4euc = np.load('2019-02-18_en_euc_angle10020.040.0.npy')
    data5euc = np.load('2019-02-18_en_euc_angle10010.050.0.npy')
    
    np.save('euc_nat_euc_angle10010.010',data1euc)
    np.save('euc_nat_euc_angle10020.020',data2euc)
    np.save('euc_nat_euc_angle10010.030',data3euc)
    np.save('euc_nat_euc_angle10020.040',data4euc)
    np.save('euc_nat_euc_angle10010.050',data5euc)
    
    data=np.concatenate((data1,data2,data3,data4,data5))
    dataeuc=np.concatenate((data1euc,data2euc,data3euc,data4euc,data5euc))

    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    
    ax.set_prop_cycle(cycler('color',['orange','darkblue'] ))
    
    bins=np.linspace(0.,90.,18.,endpoint=False)
    #trialnumber=float(len(data))
    ax.hist([data,dataeuc], bins,normed=0.)
    xmin=0.
    xmax=90.
    ymin=0.
    ymax=250.
    deltax=xmax-xmin
    deltay=ymax-ymin
    xrange=np.array([xmin,xmax]) 
    yrange=np.array([ymin,ymax])
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)
    x_ticks = np.array((xmin+0.*deltax,xmin+0.5*deltax,xmin+1.0*deltax))
    y_ticks = np.array((ymin+0.*deltay,ymin+0.5*deltay,ymin+1.0*deltay))
    trialnumber=float(len(data))
    ax.set_yticklabels(y_ticks/trialnumber)
    ax.xaxis.set_ticks(x_ticks)
    ax.yaxis.set_ticks(y_ticks)
    
    ax.set_xlabel("angle Euclidean metric [deg]")
    ax.set_ylabel("fraction of samples")
    
    pass

plt.figure(0)
ax=plt.axes()
plot_fisher_angles(ax)

plt.figure(1)
ax=plt.axes()
plot_euc_angles(ax)