#!/usr/bin/env python
from tkinter.ttk import Style
import numpy as np
import pandas as pd
import matplotlib.ticker as mtick
from matplotlib.ticker import AutoMinorLocator, LogLocator, AutoLocator
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from scipy.interpolate import make_interp_spline,BSpline

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)

def defineplot():
    # Infinite plotting stuff from down. Might not require cleaning.
    plt.rcParams['text.latex.preamble'] = [
        # i need upright \micro symbols, but you need...
        r'\usepackage{siunitx}',
        # ...this to force siunitx to actually use your fonts
        r'\usepackage[T1]{fontenc}',
        r'\sisetup{detect-all}',
        r'\usepackage{helvet}',  # set the normal font here
        # load up the sansmath so that math -> helvet##
        r'\usepackage[eulergreek,EULERGREEK]{sansmath}',
        r'\sansmath',  # <- tricky! -- gotta actually tell tex to use!
    ]

    # plt.rcParams['mathtext.fallback_to_cm'] = 'True'
    #plt.rc('font', family='sans-serif')
    #plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    # plt.rc('text', usetex=True)
    plt.rcParams['lines.antialiased'] = True
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['legend.fancybox'] = False
    plt.rcParams['legend.loc'] = 'best'
    plt.rcParams['legend.numpoints'] = 1
    plt.rcParams['legend.fontsize'] = '10'
    plt.rcParams['legend.framealpha'] = None
    plt.rcParams['legend.scatterpoints'] = 1
    plt.rcParams['legend.edgecolor'] = 'inherit'

    plt.rcParams['xtick.major.size'] = 7
    plt.rcParams['xtick.major.width'] = 0.5
    plt.rcParams['xtick.minor.size'] = 4
    plt.rcParams['xtick.minor.width'] = 0.5
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.major.size'] = 7
    plt.rcParams['ytick.major.width'] = 0.5
    plt.rcParams['ytick.minor.size'] = 4
    plt.rcParams['ytick.minor.width'] = 0.5
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['hatch.linewidth'] = 0.5 



def plot_non_equ_average():

    defineplot()

    fig, axes= plt.subplots(1, 3, figsize=(9,2.5))

    # axes[0].set_yscale('log')
    #axes[0].set_ylim((-12, -4))
    #axes[0].set_xlim((0, 3))
    #axes[0].xaxis.set_minor_locator(AutoMinorLocator(2))

    #axes[0].yaxis.set_minor_locator(AutoMinorLocator(2))
    #axes[0].tick_params(which='both', direction='in', top=True, right=True)
    #axes[0].xaxis.set_minor_locator(AutoMinorLocator(2))
    #axes[0].xaxis.set_label_position('top') 
    #axes[0].xaxis.set_tick_params(labeltop='on')
    

    
    mec = ['#232f5d','#faba64','#c91430' ]
    mfc = ['#232f5d','#faba64','#c91430' ]
    mec_exp = ['#0099d5','#1a8344','#d31372']
    mfc_exp = ['#91d0dc','#a2ce95','#f6d9e0']
    mfc_exp_new = ['#7da9da','#f7f1a4','#ea4d4b']
    mec_exp_new = ['#232f5d','#6f642b','#5c1531']
    markers = ['o', 's', '^','*']
    marker_size = 6
    line_width = 0.5
    mew = 0.75
    line_style = 'dashed'

    # -------------
    # KMCPY
    # plotting time per pass with different cell size with and without numba

    #axes[0].plot(df.x, np.log10(df.D_J), marker=markers[i], color=mec[i],mfc=mfc[i],mec=mec[i], label=str(T)+' K', ms=marker_size, lw=line_width, ls=line_style,mew=mew)
    run_time_numba=[]
    with open("supercell_scalability_log.txt") as f:
        line=f.readline()
        cell_and_time=line.replace("(","").split(")")
        lim=len(cell_and_time)
        print(cell_and_time)
        for time_ in cell_and_time[:-1]:
            print(time_.split(","))
            run_time_numba.append(float(time_.split(",")[-1].replace(" ","")))
    print(run_time_numba)
    lim=len(run_time_numba)
    run_time_no_numba=[]
    
    with open("supercell_scalability_no_numba_log.txt") as f:
        line=f.readline()
        cell_and_time=line.replace("(","").split(")")

        for time_ in cell_and_time[:-1]:
            print(time_.split(","))
            run_time_no_numba.append(float(time_.split(",")[-1].replace(" ","")))
    print(run_time_no_numba)
        
    total_cells=[]
    for i in range(1,lim+1):
        total_cells.append(i**3)
    total_cells=np.array(total_cells)
    print("total cells:",total_cells)
    def poly1(x,a1,a2,a3):
        return a1*(x**a2)+a3

    def poly2(x,b1,b2,b3):
        return b1*(x**b2)+b3
    numba,_=curve_fit(poly1,total_cells,run_time_numba,maxfev=20000)
    a1,a2,a3=numba
    fit1=poly1(total_cells,a1,a2,a3)
    r2_1=r2_score(run_time_numba,fit1)
    r2_1=round(r2_1,5)

    no_numba,_=curve_fit(poly2,total_cells,run_time_no_numba,maxfev=20000)
    b1,b2,b3=no_numba
    fit2=poly2(total_cells,b1,b2,b3)
    r2_2=r2_score(run_time_no_numba,fit2)
    r2_2=round(r2_2,5)

    xnew=np.linspace(total_cells.min(),total_cells.max(),200)

    spl_numba=make_interp_spline(total_cells,fit1,k=3)
    smooth_numba=spl_numba(xnew)

    spl_no_numba=make_interp_spline(total_cells,fit2,k=3)
    smooth_no_numba=spl_no_numba(xnew)

    a1=round(a1,4)
    a2=round(a2,4)
    a3=round(a3,4)
    b1=round(b1,4)
    b2=round(b2,4)
    b3=round(b3,4)
    axes[0].scatter(total_cells,run_time_numba,c="tab:red",marker="^")
    axes[0].scatter(total_cells,run_time_no_numba,c="tab:red",marker="v")
    axes[0].plot(xnew,smooth_numba,label="Time/pass",color="tab:red")
    axes[0].plot(xnew,smooth_no_numba,label=" ",color="tab:red",linestyle="dashed")
    
    # axes[0].plot([10,11],[-100,-100],color="tab:blue",label="Time/step")# dummy for legend

    # axes[0].plot([10,11],[-100,-100],linestyle="dashed",color="tab:blue",label=" ")
    axes[0].xaxis.set_minor_locator(AutoMinorLocator(2))

    xticks=list(total_cells.copy())
    xticks.pop(1)
    xticks.pop(1)
    xticks.pop(2)
    #xticks.pop(2)
    #print(xticks)
    #axes[0].set_xticks(xticks, xticks)
    #axes[0].set_title("Model Size")
    axes[0].set_xlabel("Cell size")
    axes[0].set_ylabel("Run time per kMC pass (s)")
    axes[0].set_ylim(0,200)
    axes[0].set_xlim(0,1000)


    #axes[0].legend(loc='best',ncol=1)
    # l = axes[0].legend(loc="upper left",bbox_to_anchor=(0,0.99),labelspacing=-0.3)
    # l.get_frame().set_linewidth(0.75)

    #-------------
    # per step time
    axes02=axes[0].twinx()
    #axes02.scatter(total_cells,np.array(run_time_numba)/np.array(total_cells),c="tab:red")
    axes02.set_yscale('log')
    axes02.set_ylim(1e-3,1e-0)
    #axes02.scatter(total_cells,np.array(run_time_no_numba)/np.array(total_cells),c="tab:blue")
    axes02.plot(total_cells,np.array(run_time_numba)/np.array(total_cells),color="tab:blue")
    axes02.plot(total_cells,np.array(run_time_no_numba)/np.array(total_cells),color="tab:blue",linestyle="dashed")
    #axes02.legend(loc="upper left")
    # xticks=list(total_cells.copy())
    # xticks.pop(1)
    # xticks.pop(1)
    # xticks.pop(2)
    # #xticks.pop(2)
    #axes02.set_xticks(xticks, xticks)
    #axes02.set_title("Comparison")

    axes02.set_xlabel("Cell size")
    axes02.set_ylabel("Run time per kMC step (s)")


    # twiny does not match the original x data! 
    """
    axes03=axes[0].twiny()
    axes03.set_xlim(0,12000)
    axes03.set_xticks([10,100,500,10000],[10,100,500,1000])
    #axes03.scatter(100,-100)# dummy
    axes03.set_xlabel("Number of events")
    #axes03.set_xscale("log")
    """
    
    #------------------------------------------------------------------
    # plot the per pass time with cluster and orbit

    #axes02.legend(loc='best',ncol=1)
    cluster_dict={}
    orbit_dict={}
    
    # for fitting
    unique_clusters=[]# unique
    runtime_list=[]# unique
    unique_orbits=[]# unique
    
    # for scatter
    clusters=[]# duplicate
    orbits=[]# duplicate
    runtime_cutoff=[] # duplicate
    
    
    
    with open("cutoff_time.log") as f:
        for line in f.readlines():
            line=line.replace("run time for ","").replace(" clusters and ",",").replace(" orbits , cutoff ",",").replace(" A is ",",").replace(" \n","").split(",")
            print(line)
            clusters.append(int(line[0]))
            orbits.append(int(line[1]))
            runtime_cutoff.append(float(line[3]))
            
            
            cluster=int(line[0])
            orbit=(int(line[1]))
            this_runtime=float(line[3])
                    
            if cluster not in cluster_dict:
                cluster_dict[cluster]=[this_runtime]
            else:
                cluster_dict[cluster].append(this_runtime)
            if orbit not in orbit_dict:
                orbit_dict[orbit]=[this_runtime]
            else:
                orbit_dict[orbit].append(this_runtime)
    for i in cluster_dict:
        unique_clusters.append(i)
        runtime_list.append(sum(cluster_dict[i])/len(cluster_dict[i]))# avg runtime   
    for i in orbit_dict:
        unique_orbits.append(i)
        
        
    # axes[2].scatter(orbits,runtime_cutoff,c="tab:blue")

    def poly3(x,a1,a2,a3):
        return a1*(x**a2)+a3

    cutoff,_=curve_fit(poly3,np.array(unique_clusters),np.array(runtime_list),maxfev=20000)
    print("for cutoff",cutoff)
    fit3=poly3(unique_clusters,*cutoff)
    r2_3=r2_score(runtime_list,fit3)
    r2_3=round(r2_3,5)
    print("r2 score for cutoff :",r2_3)

    cutoff_xnew=np.linspace(min(clusters),max(clusters),200)

    spl_cutoff=make_interp_spline(unique_clusters,fit3,k=3)
    smooth_cutoff=spl_cutoff(cutoff_xnew)

    axes[2].plot(cutoff_xnew/1000,smooth_cutoff,color="tab:red",label="Time/pass")
    axes[2].scatter(np.array(clusters)/1000,runtime_cutoff,c="tab:red",s=5)

    axes[2].set_xlim(0,25)
    axes[2].set_xticks(np.array([0,5000,10000,15000,20000,25000])/1000)
    axes[2].set_ylim(0,70)
    # axes[2].set_xscale("log")
    # axes[2].set_yscale("log")
    axes[2].set_ylabel("Run Time per kMC pass (s)")
    axes[2].set_xlabel("Number of clusters")
    #axes[2].set_title("model complexity")
    
    # axes1_orbit=axes[2].twiny()

    # axes1_orbit.set_xticks(np.linspace(min(orbits),max(orbits),7,dtype=np.int64),np.linspace(min(orbits),max(orbits),7,dtype=np.int64))
    # axes1_orbit.set_xlim(min(orbits),max(orbits))
    # # axes1_orbit.set_xscale("log")
    # axes1_orbit.set_xlabel("Number of Orbits")
    # # l = axes[2].legend(loc="best")
    # l.get_frame().set_linewidth(0.75)


    with open("basis_set_time.txt") as f:
        basis_set_time_raw=f.readline().replace("\n","").split(",")

    
    basis_set_size=[12*i+12 for i in range(0,len(basis_set_time_raw))]
    run_time=basis_set_time_raw
    def eq3(x,z1,z2,z3):
        return z2*x**z1 + z3


    popt3,_=curve_fit(eq3,basis_set_size,run_time,maxfev=5000)
    print("for basis size:", popt3)
    axes[1].plot(basis_set_size,run_time,'.-',mec='red',c='red',label="Time/pass")
    axes[1].set_ylim(0,30)
    axes[1].set_yticks(np.linspace(0,30,6),np.linspace(0,30,6))
    axes[1].set_xlim(0,400)
    axes[1].xaxis.set_minor_locator(AutoMinorLocator(2))
    axes[1].set_xlabel("Coupling strength between events")
    axes[1].set_ylabel("Run time per kMC pass (s)")
    # l = axes[1].legend(loc="best",)
    # l.get_frame().set_linewidth(0.75)



    # axes[0].text(-0.1, 1.1, "a", transform=axes[0].transAxes, 
    #         size=20, weight='bold')
    # axes[2].text(-0.1, 1.1, "c", transform=axes[2].transAxes, 
    #         size=20, weight='bold')
    # axes[1].text(-0.1, 1.1, "b", transform=axes[1].transAxes, 
    #         size=20, weight='bold')

    #fig.suptitle('Runtime on different cell size', fontsize=16)
    axes[0].xaxis.set_minor_locator(AutoMinorLocator(2))
    axes[1].xaxis.set_minor_locator(AutoMinorLocator(2))
    axes[2].xaxis.set_minor_locator(AutoMinorLocator(2))

    axes[0].tick_params(which='both', direction='in', top=True, right=False)
    axes[1].tick_params(which='both', direction='in', top=True, right=True)
    axes[2].tick_params(which='both', direction='in', top=True, right=True)

    axes[0].yaxis.set_minor_locator(AutoMinorLocator(2))
    axes[1].yaxis.set_minor_locator(AutoMinorLocator(2))
    axes[2].yaxis.set_minor_locator(AutoMinorLocator(2))

    axes[2].yaxis.set_major_locator(AutoLocator())
    fig.tight_layout()
    fig.savefig('scalability.pdf')


# plot(293)
plot_non_equ_average()
