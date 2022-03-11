#!/usr/bin/env python2
# encoding: utf-8

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np







###############################
# Plot functions for subplots #
###############################
#
# naming scheme: plot_<key>(ax)
#
# ax is the Axes to plot into
    

def plot_f1(ax):
    
    ax.set_xlim([0,60])
    ymin=-0.08
    ymax=0.
    deltay=ymax-ymin
    
    yrange=np.array([ymin,ymax])
    ax.set_ylim(yrange)
    y_ticks = np.array((ymin+0.*deltay,ymin+0.5*deltay,ymin+1.*deltay))
    ax.yaxis.set_ticks(y_ticks)
    
    data_true=np.load("2020-04-26_f1.npy")
    data_rates=np.load("2020-04-26_fs_rates.npy")
    
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.scatter(data_rates,data_true,s=10,color="darkblue",clip_on=False)
    
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$g_1$")
    pass





def plot_f2(ax):
    ax.set_xlim([0,60])
    ymin=-0.02
    ymax=0.
    deltay=ymax-ymin
    
    yrange=np.array([ymin,ymax])
    ax.set_ylim(yrange)
    y_ticks = np.array((ymin+0.*deltay,ymin+0.5*deltay,ymin+1.0*deltay))
    ax.yaxis.set_ticks(y_ticks)
    
    data_true=np.load("2020-04-26_f2.npy")
    data_rates=np.load("2020-04-26_fs_rates.npy")
    
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.scatter(data_rates,data_true,s=10,color="darkblue",clip_on=False)
    
    
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$g_2$")
    pass


def plot_f3(ax):
    
    ax.set_xlim([0,60])
    ymin=-0.02
    ymax=0.
    deltay=ymax-ymin
    
    yrange=np.array([ymin,ymax])
    ax.set_ylim(yrange)
    y_ticks = np.array((ymin+0.*deltay,ymin+0.5*deltay,ymin+1.0*deltay))
    ax.yaxis.set_ticks(y_ticks)
    
    data_true=np.load("2020-04-26_f3.npy")
    data_rates=np.load("2020-04-26_fs_rates.npy")
    
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)

    ax.scatter(data_rates,data_true,s=10,color="darkblue",clip_on=False)
    
    
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$g_3$")
    pass

def plot_f4(ax):
    
    ax.set_xlim([0,60])
    ymin=0.
    ymax=0.08
    deltay=ymax-ymin
    
    yrange=np.array([ymin,ymax])
    ax.set_ylim(yrange)
    y_ticks = np.array((ymin+0.*deltay,ymin+0.5*deltay,ymin+1.0*deltay))
    ax.yaxis.set_ticks(y_ticks)
    
    data_true=np.load("2020-04-26_f4.npy")
    data_rates=np.load("2020-04-26_fs_rates.npy")
    
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    
    ax.scatter(data_rates,data_true,s=10,color="darkblue",clip_on=False)
    ax.scatter(np.mean(data_rates,axis=1),np.mean(data_true,axis=1),s=5,color="orange",marker="*",clip_on=False)
    ax.legend(("samples","mean"),fontsize=8.,handlelength=0.5,frameon=False,loc=(0.05,0.6))
    
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$g_4$")
    pass


def plot_gamma_w_cst(ax):
    xmin=-3.
    xmax=3.
    ymin=-3.
    ymax=3.
    deltax=xmax-xmin
    deltay=ymax-ymin
    xrange=np.array([xmin,xmax]) 
    yrange=np.array([ymin,ymax])
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)
    x_ticks = np.array((ymin+0.*deltax,xmin+0.5*deltax,xmin+1.0*deltax))
    ax.set_xticklabels((r"$-3.00$",r"$0.00$",r"$3.00$"))
    y_ticks = np.array((ymin+0.*deltay,ymin+0.5*deltay,ymin+1.0*deltay))
    ax.set_yticklabels((r"$-3.00$",r"$0.00$",r"$3.00$"))
    ax.xaxis.set_ticks(x_ticks)
    ax.yaxis.set_ticks(y_ticks)
    
    data_true=np.load("2020-04-26_gammaw_exact.npy")
    data_approx=np.load("2020-04-26_gammaw_approx_cst0.04.npy")
    
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    
    ax.plot(np.ndarray.flatten(data_approx),np.ndarray.flatten(data_approx),color="orange")
    ax.scatter(data_approx,data_true,s=10,color="darkblue")
    
    ax.legend(("bisecting","samples"),fontsize=8.,handlelength=0.5,frameon=False,loc=(0.5,0.05))
    

    
    ax.set_xlabel(r"$c_{\rm p} V$")
    ax.set_ylabel(r"$\gamma_{\rm w}$")
    pass



def plot_f1_approx(ax):
    data_true=np.load("2020-05-28_f1Nr.npy")
    data_approx=np.load("2020-05-28_f1approx.npy")
    data_N=np.load("2020-05-28_N_Nr.npy")
    
    ax.set_xlim([0,10000])
    ax.xaxis.set_ticks((0,5000))
    ymin=-0.9
    ymax=0.
    deltay=ymax-ymin
    
    yrange=np.array([ymin,ymax])
    ax.set_ylim(yrange)
    y_ticks = np.array((ymin+0.*deltay,ymin+0.5*deltay,ymin+1.*deltay))
    ax.yaxis.set_ticks(y_ticks)
    
    
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    
    
    ax.scatter(data_N,data_true,s=10,color="darkblue",clip_on=False)
    ax.scatter(data_N,data_approx,s=5,color="orange",marker='*',clip_on=False)
    ax.legend(("samples","approximation"),fontsize=8.,handlelength=0.5,frameon=False,loc=(0.2,0.05))
    
    #ax1.spines['top'].set_visible(False)
    ax.set_xlabel(r"$n*r$")
    ax.set_ylabel(r"$g_1$")
    #ax1.set_ylabel("approx",labelpad=-7.)
    
    
def plot_f1N(ax):
    ax.set_xlim([0,250])
    
    ymin=-0.24
    ymax=0.
    deltay=ymax-ymin
    
    yrange=np.array([ymin,ymax])
    ax.set_ylim(yrange)
    y_ticks = np.array((ymin+0.*deltay,ymin+0.5*deltay,ymin+1.0*deltay))
    ax.yaxis.set_ticks(y_ticks)
    data_true=np.load("2020-04-26_f1N.npy")
    data_rates=np.load("2020-04-26_fs_N.npy")
    
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.scatter(data_rates,data_true,s=10,color="darkblue",clip_on=False)
    
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$g_1$")
    pass





def plot_f2N(ax):
    ax.set_xlim([0,250])
    
    ymin=-0.04
    ymax=0.
    deltay=ymax-ymin
    
    yrange=np.array([ymin,ymax])
    ax.set_ylim(yrange)
    y_ticks = np.array((ymin+0.*deltay,ymin+0.5*deltay,ymin+1.0*deltay))
    ax.yaxis.set_ticks(y_ticks)
    
    data_true=np.load("2020-04-26_f2N.npy")
    data_rates=np.load("2020-04-26_fs_N.npy")
    
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.scatter(data_rates,data_true,s=10,color="darkblue",clip_on=False)
    
    
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$g_2$")
    pass


def plot_f3N(ax):
    
    ax.set_xlim([0,250])
    
    ymin=-0.04
    ymax=0.
    deltay=ymax-ymin
    
    yrange=np.array([ymin,ymax])
    ax.set_ylim(yrange)
    y_ticks = np.array((ymin+0.*deltay,ymin+0.5*deltay,ymin+1.0*deltay))
    ax.yaxis.set_ticks(y_ticks)
    data_true=np.load("2020-04-26_f3N.npy")
    data_rates=np.load("2020-04-26_fs_N.npy")
    
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.scatter(data_rates,data_true,s=10,color="darkblue",clip_on=False)
    
    
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$g_3$")
    pass

def plot_f4N(ax):
    
    ax.set_xlim([0,250])
    
    ymin=0.
    ymax=0.08
    deltay=ymax-ymin
    
    yrange=np.array([ymin,ymax])
    ax.set_ylim(yrange)
    y_ticks = np.array((ymin+0.*deltay,ymin+0.5*deltay,ymin+1.0*deltay))
    ax.yaxis.set_ticks(y_ticks)
    data_true=np.load("2020-04-26_f4N.npy")
    data_rates=np.load("2020-04-26_fs_N.npy")
    
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
        
    
    ax.scatter(data_rates,data_true,s=10,color="darkblue",clip_on=False)
    ax.scatter(np.mean(data_rates,axis=1),np.mean(data_true,axis=1),s=5,color="orange",marker="*",clip_on=False)
    ax.legend(("samples","mean"),fontsize=8.,handlelength=0.5,frameon=False,loc=(0.05,0.6))
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$g_4$")
    pass
    
    
    
def plot_gamma_u_approx(ax):
      
    xmin=0.
    xmax=0.06
    ymin=0.
    ymax=0.06
    deltax=xmax-xmin
    deltay=ymax-ymin
    xrange=np.array([xmin,xmax]) 
    yrange=np.array([ymin,ymax])
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)
    x_ticks = np.array((ymin+0.*deltax,xmin+0.5*deltax,xmin+1.0*deltax))
    y_ticks = np.array((ymin+0.*deltay,ymin+0.5*deltay,ymin+1.0*deltay))
    ax.xaxis.set_ticks(x_ticks)
    ax.yaxis.set_ticks(y_ticks)
    data_true=np.load("2020-04-26_gammau_exact.npy")
    data_approx=np.load("2020-04-26_gammau_approx.npy")
    
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.scatter(data_approx,data_true,s=10,color="darkblue",clip_on=False)
    ax.plot(np.ndarray.flatten(data_approx),np.ndarray.flatten(data_approx),color="orange")
    
    ax.legend(("bisecting","samples"),fontsize=8.,handlelength=0.5,frameon=False,loc=(0.5,0.05))
    
    ax.set_xlabel(r"$s$",fontsize=8.)
    ax.set_ylabel(r"$\gamma_{\rm u}$")
    
    pass

def plot_gamma_w_approx(ax):
    xmin=-3.
    xmax=3.
    ymin=-3.
    ymax=3.
    deltax=xmax-xmin
    deltay=ymax-ymin
    xrange=np.array([xmin,xmax]) 
    yrange=np.array([ymin,ymax])
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)
    x_ticks = np.array((ymin+0.*deltax,xmin+0.5*deltax,xmin+1.0*deltax))
    ax.set_xticklabels((r"$-3.00$",r"$0.00$",r"$3.00$"))
    y_ticks = np.array((ymin+0.*deltay,ymin+0.5*deltay,ymin+1.0*deltay))
    ax.set_yticklabels((r"$-3.00$",r"$0.00$",r"$3.00$"))
    ax.xaxis.set_ticks(x_ticks)
    ax.yaxis.set_ticks(y_ticks)
    data_true=np.load("2020-04-26_gammaw_exact.npy")
    data_approx=np.load("2020-04-26_gammaw_approx.npy")
    
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.scatter(data_approx,data_true,s=10,color="darkblue")
    ax.plot(np.ndarray.flatten(data_approx),np.ndarray.flatten(data_approx),color="orange")
    
    ax.legend(("bisecting","samples"),fontsize=8.,handlelength=0.5,frameon=False,loc=(0.5,0.05))
    

    
    ax.set_xlabel(r"$g_4 V$")
    ax.set_ylabel(r"$\gamma_{\rm w}$")
    pass


def plot_gamma_s_nl(ax):
    data_true = np.load('2020-04-26_gs.npy')
   
    data_vm=np.load('2020-04-26_m.npy')
    data_nl=np.load('2020-04-26_phi_m.npy')
    ax.set_xlim([-20.,20.])
   
    ymin=0.
    ymax=20.
    deltay=ymax-ymin
    
    yrange=np.array([ymin,ymax])
    ax.set_ylim(yrange)
    y_ticks = np.array((ymin+0.*deltay,ymin+0.5*deltay,ymin+1.0*deltay))
    ax.yaxis.set_ticks(y_ticks)
    
    ax1=ax.twinx()    
    y1min=0.
    y1max=100.
    deltay1=y1max-y1min
    
    y1range=np.array([ymin,ymax])
    ax1.set_ylim(y1range)
    y1_ticks = np.array((y1min+0.*deltay1,y1min+0.5*deltay1,y1min+1.0*deltay1))
    ax1.yaxis.set_ticks(y1_ticks)
    
    

    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.scatter(data_vm,data_true,s=10,color='darkblue')
   
    ax1.scatter(data_vm,data_nl,s=10,color='orange')
    
    
    ax1.spines['top'].set_visible(False)
    ax.set_xlabel(r"$\mu_{\rm v}$")
    ax.set_ylabel(r"$\gamma_{\rm s}$")
    ax1.set_ylabel(r"$\phi(\mu_{\rm v})\ \mathrm{[Hz]}$")
    
    pass

fig=plt.figure(0)
ax=plt.axes()
plot_f1(ax)

fig1=plt.figure(1)
ax=plt.axes()
plot_f2(ax)

fig2=plt.figure(2)
ax=plt.axes()
plot_f3(ax)

fig3=plt.figure(3)
ax=plt.axes()
plot_f4(ax)

fig4=plt.figure(4)
ax=plt.axes()
plot_f1N(ax)

fig5=plt.figure(5)
ax=plt.axes()
plot_f2N(ax)

fig6=plt.figure(6)
ax=plt.axes()
plot_f3N(ax)

fig7=plt.figure(7)
ax=plt.axes()
plot_f4N(ax)

fig8=plt.figure(8)
ax=plt.axes()
plot_f1_approx(ax)

fig9=plt.figure(9)
ax=plt.axes()
plot_gamma_u_approx(ax)

fig10=plt.figure(10)
ax=plt.axes()
plot_gamma_w_approx(ax)

fig11=plt.figure(11)
ax=plt.axes()
plot_gamma_w_cst(ax)

fig12=plt.figure(12)
ax=plt.axes()
plot_gamma_s_nl(ax)


