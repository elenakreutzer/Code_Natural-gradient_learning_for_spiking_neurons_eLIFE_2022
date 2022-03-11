# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 14:35:14 2022

@author: Elena
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import File01_simulation_parameters as para
from cycler import cycler


def plot_spikes_before(ax,**parameter):
    
    
    data = np.load('2019-02-02_output_spikes_psth_start.npy')
    
    ax.scatter(np.nonzero(data)[1]*parameter["dt"],np.nonzero(data)[0],color="indianred",s=0.005)
    np.save("spikes_before_time",np.nonzero(data)[1]*parameter["dt"])
    np.save("spikes_before_neuron",np.nonzero(data)[0])
    ax.set_xticks([])
    
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    
    ax.set_xlabel(r'$t$ [s]')
    ax.set_ylabel("PSTH")

    pass



    
def plot_spikes_after(ax,**parameter):
    
    
    data = np.load('2019-02-02_output_spikes_psth_end.npy')

    print(np.shape(data))
    ax.scatter(np.nonzero(data)[1]*parameter["dt"],np.nonzero(data)[0],color="indianred",s=0.005)
    np.save("spikes_after_time",np.nonzero(data)[1]*parameter["dt"])
    np.save("spikes_after_neuron",np.nonzero(data)[0])
    ax.set_xticks([])
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    
    ax.set_xlabel(r'$t$ [s]')
    ax.set_ylabel(r'$\#$ trial')

    
    pass

def plot_spikes_before_teach(ax,**parameter):
    
    data_teach=np.load("2019-02-02_output_spikes_psth_teacher.npy")
    print(np.shape(data_teach))
    
    ax.scatter(np.nonzero(data_teach)[1]*parameter["dt"],np.nonzero(data_teach)[0],color="orange",s=0.01)
    np.save("spikes_teach_time",np.nonzero(data_teach)[1]*parameter["dt"])
    np.save("spikes_teach_neuron",np.nonzero(data_teach)[0])
    
    
    ax.set_xticks([])
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    
    #ax.set_xlabel(core.x_labels["spikes_teach"])
    ax.set_ylabel("PSTH")


    pass



    
def plot_spikes_after_teach(ax, **parameter):


    
    data_teach=np.load("2019-02-02_output_spikes_psth_teacher.npy")
    ax.set_xticks([])
    ax.scatter(np.nonzero(data_teach)[1]*parameter["dt"],np.nonzero(data_teach)[0],color="orange",s=0.01)
   
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    
    #ax.set_xlabel(core.x_labels["spikes_teach"])
    #ax.set_ylabel(core.y_labels["spikes_teach"])

    
    pass

def plot_histo_before(ax,**parameter):
    
    data = np.load('2019-02-02_output_spikes_psth_start.npy')
    data_teach=np.load("2019-02-02_output_spikes_psth_teacher.npy")
    spiketimes=np.nonzero(data)[1]*parameter["dt"]
    spiketimes_teach=np.nonzero(data_teach)[1]*parameter["dt"]
    np.save("spiketimes_histo_before",spiketimes)
    np.save("spiketimes_histo_teach",spiketimes_teach)
    
    ax.set_prop_cycle(cycler('color',["indianred","orange"] ))
    
    bins=np.linspace(0.,0.25,20,endpoint=False)
    trialnumber=float(len(data_teach[:,0]))
    print(trialnumber)
    ax.hist([spiketimes,spiketimes_teach], bins, label=['x', 'y'],normed=0.)
    ax.set_xticks([])
    ax.set_yticks([300,])
    ytickslabel=(300./(trialnumber),)
    ax.set_yticklabels(ytickslabel)

    
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_xlabel(r'$t$ [s]')
    ax.set_ylabel('spikes/trial',labelpad=10.)
    
    pass

def plot_histo_after(ax,**parameter):
    
    data = np.load('2019-02-02_output_spikes_psth_end.npy')
    data_teach=np.load("2019-02-02_output_spikes_psth_teacher.npy")
    spiketimes=np.nonzero(data)[1]*parameter["dt"]
    np.save("spiketimes_histo_after",spiketimes)
    spiketimes_teach=np.nonzero(data_teach)[1]*parameter["dt"]
    #print(np.shape(spiketimes), np.shape(spiketimes_teach))
    trialnumber=float(len(data_teach[:,0]))
    
    
    ax.set_prop_cycle(cycler('color',["indianred","orange"] ))
    
    bins=np.linspace(0.,0.25,20,endpoint=False)
    ax.hist([spiketimes,spiketimes_teach], bins, label=['x', 'y'],normed=0.)
    ax.set_xticks([])
    ax.set_yticks([300,])
    ytickslabel=(300./(trialnumber),)
    ax.set_yticklabels(ytickslabel)

    
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    
    ax.set_xlabel(r'$t$ [s]')
    ax.set_ylabel('spikes/trial',labelpad=10.)
    
    pass

def plot_voltage_before(ax, **parameter):
    data = np.load('2019-02-02_membrane_potential_psth_start.npy')[10]
    data_teach=np.load("2019-02-02_membrane_potential_psth_teacher.npy")[10]

    #ax.set_xlim([0.,.5])
    
    #subtract resting potential
    data=data-70.
    data_teach=data_teach-70.
    
    time=np.arange(0.,500,1.)*parameter["dt"]
    np.save("voltage_before",data)
    np.save("voltage_teach",data_teach)
    np.save("voltage_time",time)
    
    ax.set_prop_cycle(cycler('color',["indianred","orange"] ))
    
    ax.plot(time,data,color="indianred",linewidth=1.)
    ax.plot(time,data_teach,color="orange",linewidth=1.)
    ax.set_xticks([0.1,0.2])


    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    #plotf.trace_plot(data)
    ax.set_xlabel(r'$t$ [s]')
    ax.set_ylabel(r'$V$ [mV]')
    #ax.legend(("student","teacher"),loc=0,fontsize=8.,labelspacing=0.1,borderaxespad=0.1,borderpad=0.,handlelength=0.5,frameon=False)
    

    pass

def plot_voltage_after(ax,**parameter):
    data = np.load('2019-02-02_membrane_potential_psth_end.npy')[10]
    data_teach=np.load("2019-02-02_membrane_potential_psth_teacher.npy")[10]
    #time_params=np.load("time_parameters.npy").item()
    #ax.set_xlim([0.,.5])

    time=np.arange(0,500)*parameter["dt"]
    np.save("voltage_after",data)
    #subtract resting potential
    data=data-70.
    data_teach=data_teach-70.

    ax.plot(time,data,color="indianred",linewidth=1.)
    ax.plot(time,data_teach,color="orange",linewidth=1.)
    ax.set_xticks([0.1,0.2])
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    #plotf.trace_plot(data)
    ax.set_xlabel(r'$t$ [s]')
    #ax.set_ylabel(r'$V$ [mV]')
   #ax.legend(("student","teacher"),loc=0,fontsize=8.,labelspacing=0.1,borderaxespad=0.1,borderpad=0.,handlelength=0.5,frameon=False)
    


    pass




plt.figure(0)
ax=plt.axes()
plot_spikes_before(ax,**para.parameter)

plt.figure(1)
ax=plt.axes()
plot_spikes_after(ax,**para.parameter)

plt.figure(2)
ax=plt.axes()
plot_spikes_before_teach(ax,**para.parameter)

plt.figure(3)
ax=plt.axes()
plot_spikes_after_teach(ax,**para.parameter)

plt.figure(4)
ax=plt.axes()
plot_voltage_before(ax,**para.parameter)

plt.figure(5)
ax=plt.axes()
plot_voltage_after(ax,**para.parameter)

plt.figure(6)
ax=plt.axes()
plot_histo_before(ax,**para.parameter)

plt.figure(7)
ax=plt.axes()
plot_histo_after(ax,**para.parameter)
