# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:46:23 2024

@author: amaes
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import scipy.optimize
from matplotlib.lines import Line2D

def switching_distributions(model, dataset_name):
    
    dataset_path='data/'
    # read from pickle files
    file = open(dataset_path+'actions_'+dataset_name+'.pickle', 'rb')
    actions = pickle.load(file)
    file.close()
    file = open(dataset_path+'states_'+dataset_name+'.pickle', 'rb')
    states = pickle.load(file)
    file.close()
    file = open(dataset_path+'episode_ends_'+dataset_name+'.pickle', 'rb')
    episode_ends = pickle.load(file) # Marks one-past the last index for each episode
    file.close()

    #phase A
    nb_A = 7
    locs_A_left = []
    synth_locs_A_left = []
    locs_A_right = []
    synth_locs_A_right = []
    for i in range(nb_A):
        file = open('data_synthetic/'+model+'/states_synthetic_'+str(i)+'.pickle', 'rb')
        states_synth = pickle.load(file) 
        file.close()
        locs = states_synth[:,0]<40
        idx = np.where(locs[1:]!=locs[0:-1] )[0]/3
        if locs[0]:
            synth_locs_A_left.append(idx[0])
            for k in range(1,len(idx)-1,2):
                synth_locs_A_left.append(idx[k+1]-idx[k])
                synth_locs_A_right.append(idx[k]-idx[k-1])
        else:
            synth_locs_A_right.append(idx[0])
            for k in range(1,len(idx)-1,2):
                synth_locs_A_right.append(idx[k+1]-idx[k])
                synth_locs_A_left.append(idx[k]-idx[k-1])

        
        start_idx = 0
        if i>0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        locs = states[start_idx:end_idx,0]<40
        idx = np.where(locs[1:]!=locs[0:-1] )[0]/3
        if locs[0]:
            locs_A_left.append(idx[0])
            for k in range(1,len(idx)-1,2):
                locs_A_left.append(idx[k+1]-idx[k])
                locs_A_right.append(idx[k]-idx[k-1])
        else:
            locs_A_right.append(idx[0])
            for k in range(1,len(idx)-1,2):
                locs_A_right.append(idx[k+1]-idx[k])
                locs_A_left.append(idx[k]-idx[k-1])
                
    sides_A = []            
    synth_sides_A = []
    for i in range(5):
        synth_sides_A.append(synth_locs_A_left)
        synth_sides_A.append(synth_locs_A_right)
        sides_A.append(locs_A_left)
        sides_A.append(locs_A_right)


    concentrations = np.array([8,10,9,9,7])
    #phase B
    sides_B = []
    synth_sides_B = []
    for i in range(5):
        left = []
        right = []
        synth_left = []
        synth_right = []
        for j in range(concentrations[i]):
            file = open('data_synthetic/'+model+'/states_synthetic_'+str(nb_A+np.sum(concentrations[0:i])+j)+'.pickle', 'rb')
            states_synth = pickle.load(file) 
            file.close()
            locs = states_synth[:,0]<40
            idx = np.where(locs[1:]!=locs[0:-1] )[0]/3
            if locs[0] and len(idx)>0:
                synth_left.append(idx[0])
                for k in range(1,len(idx)-1,2):
                    synth_left.append(idx[k+1]-idx[k])
                    synth_right.append(idx[k]-idx[k-1])
            if locs[0] == 0 and len(idx)>0:
                synth_right.append(idx[0])
                for k in range(1,len(idx)-1,2):
                    synth_right.append(idx[k+1]-idx[k])
                    synth_left.append(idx[k]-idx[k-1])
                
            start_idx = episode_ends[nb_A+np.sum(concentrations[0:i])+j-1]
            end_idx = episode_ends[nb_A+np.sum(concentrations[0:i])+j]   
            locs = states[start_idx:end_idx,0]<40
            idx = np.where(locs[1:]!=locs[0:-1] )[0]/3
            if locs[0]:
                left.append(idx[0])
                for k in range(1,len(idx)-1,2):
                    left.append(idx[k+1]-idx[k])
                    right.append(idx[k]-idx[k-1])
            else:
                right.append(idx[0])
                for k in range(1,len(idx)-1,2):
                    right.append(idx[k+1]-idx[k])
                    left.append(idx[k]-idx[k-1])
    
        sides_B.append(np.vstack(left))
        sides_B.append(np.vstack(right))
        synth_sides_B.append(np.hstack(synth_left))
        synth_sides_B.append(np.hstack(synth_right))
    
    
    #phase C
    sides_C = []
    synth_sides_C = []
    for i in range(5):
        left = []
        right = []
        synth_left = []
        synth_right = []
        for j in range(concentrations[i]):
            file = open('data_synthetic/'+model+'/states_synthetic_'+str(nb_A+43+np.sum(concentrations[0:i])+j)+'.pickle', 'rb')
            states_synth = pickle.load(file) 
            file.close()
            locs = states_synth[:,0]<40
            idx = np.where(locs[1:]!=locs[0:-1] )[0]/3
            if locs[0] and len(idx)>0:
                synth_left.append(idx[0])
                for k in range(1,len(idx)-1,2):
                    synth_left.append(idx[k+1]-idx[k])
                    synth_right.append(idx[k]-idx[k-1])
            if locs[0] == 0 and len(idx)>0:
                synth_right.append(idx[0])
                for k in range(1,len(idx)-1,2):
                    synth_right.append(idx[k+1]-idx[k])
                    synth_left.append(idx[k]-idx[k-1])
            
            start_idx = episode_ends[nb_A+43+np.sum(concentrations[0:i])+j-1]
            end_idx = episode_ends[nb_A+43+np.sum(concentrations[0:i])+j]   
            locs = states[start_idx:end_idx,0]<40
            idx = np.where(locs[1:]!=locs[0:-1] )[0]/3
            if locs[0]:
                left.append(idx[0])
                for k in range(1,len(idx)-1,2):
                    left.append(idx[k+1]-idx[k])
                    right.append(idx[k]-idx[k-1])
            else:
                right.append(idx[0])
                for k in range(1,len(idx)-1,2):
                    right.append(idx[k+1]-idx[k])
                    left.append(idx[k]-idx[k-1])
    
        sides_C.append(np.vstack(left))
        sides_C.append(np.vstack(right))
        synth_sides_C.append(np.hstack(synth_left))
        synth_sides_C.append(np.hstack(synth_right))
    

    return synth_sides_A, synth_sides_B, synth_sides_C, sides_A, sides_B, sides_C


model = 't_801_0'
dataset = 'balanced7_x1'
sides_A, sides_B, sides_C, A_, B_, C_ = switching_distributions(model, dataset)

#PLOTS THE DWELL TIMES
colors = ['olive','olivedrab','yellowgreen']
fig, axs = plt.subplots(3, 5,sharex=False,sharey=True,)
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel("Time spent in side of cage [seconds]")
plt.ylabel("ECDF")

log_x = False
log_y = False
for i in range(5):
    sns.ecdfplot(ax=axs[0,i],data=sides_A[2*i],log_scale=(log_x,log_y),legend=False,color=colors[0])
    sns.ecdfplot(ax=axs[0,i],data=sides_A[2*i+1],log_scale=(log_x,log_y),legend=False,color=colors[0],linestyle='--')
    sns.ecdfplot(ax=axs[0,i],data=A_[2*i],log_scale=(log_x,log_y),legend=False,color='b',alpha=0.4)
    sns.ecdfplot(ax=axs[0,i],data=A_[2*i+1],log_scale=(log_x,log_y),legend=False,color='b',alpha=0.4,linestyle='--')
    axs[0,i].set_xlim(0,100)
    axs[0,i].set_ylabel('')

    sns.ecdfplot(ax=axs[1,i],data=sides_B[2*i],log_scale=(log_x,log_y),legend=False,color=colors[1])
    sns.ecdfplot(ax=axs[1,i],data=sides_B[2*i+1],log_scale=(log_x,log_y),legend=False,color=colors[1],linestyle='--')
    sns.ecdfplot(ax=axs[1,i],data=B_[2*i],log_scale=(log_x,log_y),legend=False,color='b',alpha=0.4)
    sns.ecdfplot(ax=axs[1,i],data=B_[2*i+1],log_scale=(log_x,log_y),legend=False,color='b',alpha=0.4,linestyle='--')
    axs[1,i].set_xlim(0,100)
    axs[1,i].set_ylabel('')

    sns.ecdfplot(ax=axs[2,i],data=sides_C[2*i],log_scale=(log_x,log_y),legend=False,color=colors[2])
    sns.ecdfplot(ax=axs[2,i],data=sides_C[2*i+1],log_scale=(log_x,log_y),legend=False,color=colors[2],linestyle='--')
    sns.ecdfplot(ax=axs[2,i],data=C_[2*i],log_scale=(log_x,log_y),legend=False,color='b',alpha=0.4)
    sns.ecdfplot(ax=axs[2,i],data=C_[2*i+1],log_scale=(log_x,log_y),legend=False,color='b',alpha=0.4,linestyle='--')
    axs[2,i].set_xlim((0,100))
    axs[2,i].set_ylabel('')
plt.savefig('figures/model_quality/dwell_times_'+model+'.pdf', format="pdf", bbox_inches="tight")


fig, axs = plt.subplots(1, 11,sharex=False,sharey=True,figsize=(35,5))
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel("Time spent in side of cage (s)")
plt.ylabel("ECDF")

log_x = False
log_y = False
sns.ecdfplot(ax=axs[0],data=sides_A[8],log_scale=(log_x,log_y),legend=False,color=colors[0])
sns.ecdfplot(ax=axs[0],data=sides_A[9],log_scale=(log_x,log_y),legend=False,color=colors[0],linestyle='--')
sns.ecdfplot(ax=axs[0],data=A_[8],log_scale=(log_x,log_y),legend=False,color='b',alpha=0.4)
sns.ecdfplot(ax=axs[0],data=A_[9],log_scale=(log_x,log_y),legend=False,color='b',alpha=0.4,linestyle='--')
axs[0].set_xlim(0,100)
axs[0].set_ylabel('')
for i in range(5): 
    sns.ecdfplot(ax=axs[i+1],data=sides_B[2*i],log_scale=(log_x,log_y),legend=False,color=colors[1])
    sns.ecdfplot(ax=axs[i+1],data=sides_B[2*i+1],log_scale=(log_x,log_y),legend=False,color=colors[1],linestyle='--')
    sns.ecdfplot(ax=axs[i+1],data=B_[2*i],log_scale=(log_x,log_y),legend=False,color='b',alpha=0.4)
    sns.ecdfplot(ax=axs[i+1],data=B_[2*i+1],log_scale=(log_x,log_y),legend=False,color='b',alpha=0.4,linestyle='--')
    axs[i+1].set_xlim((0,100))
    axs[i+1].set_ylabel('')

    sns.ecdfplot(ax=axs[i+6],data=sides_C[2*i],log_scale=(log_x,log_y),legend=False,color=colors[2])
    sns.ecdfplot(ax=axs[i+6],data=sides_C[2*i+1],log_scale=(log_x,log_y),legend=False,color=colors[2],linestyle='--')
    sns.ecdfplot(ax=axs[i+6],data=C_[2*i],log_scale=(log_x,log_y),legend=False,color='b',alpha=0.4)
    sns.ecdfplot(ax=axs[i+6],data=C_[2*i+1],log_scale=(log_x,log_y),legend=False,color='b',alpha=0.4,linestyle='--')
    axs[i+6].set_xlim((0,100))
    axs[i+6].set_ylabel('')
plt.savefig('figures/model_quality/dwell_times_hor_'+model+'.pdf', format="pdf", bbox_inches="tight")



#PLOTS THE PARAMETERS OF EXPONENTIAL DISTRIBUTIONS FIT TO THE SIDE DWELL TIMES
def func(x, a):
    return 1 - np.exp(-a * x)

exp_params = np.zeros((15,2))
errs = np.zeros((15,2))
for j in range(2):
    for i in range(5):
        temp = scipy.stats.ecdf(sides_A[2*i+j])
        xdata = temp.cdf.quantiles
        ydata = temp.cdf.probabilities
        popt, pcov = scipy.optimize.curve_fit(func, xdata, ydata)
        exp_params[i,j] = popt[0]
        errs[i,j] = np.sum( (ydata - func(xdata, popt[0]))**2 )
        
        temp = scipy.stats.ecdf(sides_B[2*i+j])
        xdata = temp.cdf.quantiles
        ydata = temp.cdf.probabilities
        popt, pcov = scipy.optimize.curve_fit(func, xdata, ydata)
        exp_params[i+5,j] = popt[0]
        errs[i+5,j] = np.sum( (ydata - func(xdata, popt[0]))**2 )
        
        temp = scipy.stats.ecdf(sides_C[2*i+j])
        xdata = temp.cdf.quantiles
        ydata = temp.cdf.probabilities
        popt, pcov = scipy.optimize.curve_fit(func, xdata, ydata)
        exp_params[i+10,j] = popt[0]
        errs[i+10,j] = np.sum( (ydata - func(xdata, popt[0]))**2 )
        
custom_lines = [Line2D([0], [0], color=colors[0]),
                Line2D([0], [0],color=colors[1]),
                Line2D([0], [0],color=colors[2])]

fig, ax = plt.subplots()
ax.plot(np.arange(5)+6,1./exp_params[0:5,1],color=colors[0],linestyle='--')
ax.plot(np.arange(5)+6,1./exp_params[5:10,1],color=colors[1],linestyle='--')
ax.plot(np.arange(5)+6,1./exp_params[10:15,1],color=colors[2],linestyle='--')
ax.plot(1./exp_params[0:5,0],color=colors[0])
ax.plot(1./exp_params[5:10,0],color=colors[1])
ax.plot(1./exp_params[10:15,0],color=colors[2])
ax.set_xticks(np.arange(11), ['1', '3', '10','30', '90', '', '1', '3', '10','30', '90'])
ax.set_xlabel('TMT concentration')
ax.set_ylabel('Exponential distribution parameter fit [s]')
ax.spines[['right', 'top']].set_visible(False)
ax.legend(custom_lines, ['Phase A', 'Phase B', 'Phase C'])
plt.savefig('figures/model_quality/dwell_times_'+model+'_fit.pdf', format="pdf", bbox_inches="tight")




"""
phA = []
for i in range(7):
    file = open('data_synthetic/'+model+'/states_synthetic_'+str(i)+'.pickle', 'rb')
    states_synth = pickle.load(file) 
    file.close()
    phA.append(len(np.where(states_synth[:,0]<40)[0])/states_synth.shape[0])
    

phB = []
for i in range(7,50):
    file = open('data_synthetic/'+model+'/states_synthetic_'+str(i)+'.pickle', 'rb')
    states_synth = pickle.load(file) 
    file.close()
    phB.append(len(np.where(states_synth[:,0]<40)[0])/states_synth.shape[0])
    

phC = []
for i in range(50,93):
    file = open('data_synthetic/'+model+'/states_synthetic_'+str(i)+'.pickle', 'rb')
    states_synth = pickle.load(file) 
    file.close()
    phC.append(len(np.where(states_synth[:,0]<40)[0])/states_synth.shape[0])
    

colors = ['olive','olivedrab','yellowgreen']
concentrations = np.array([8,10,9,9,7])
cms = np.cumsum(concentrations)
width = 0.4
x = np.arange(0,10,2)
# plot data in grouped manner of bar type 
fig, ax = plt.subplots()
yA = np.zeros((5,))
yB = np.zeros((5,))
yC = np.zeros((5,))
for i in range(5):
    ax.scatter(x[i]-0.5 + 0.3*np.random.random(7)  - 0.15, phA, color=colors[0])
    yA[i] = np.mean(phA)
    ax.scatter(x[i] + 0.3*np.random.random(concentrations[i])  - 0.15, phB[cms[i]-concentrations[i]:cms[i]], color=colors[1])
    yB[i] = np.mean(phB[cms[i]-concentrations[i]:cms[i]])
    ax.scatter(x[i]+0.5 + 0.3*np.random.random(concentrations[i])  -0.15, phC[cms[i]-concentrations[i]:cms[i]], color=colors[2])
    yC[i] = np.mean(phC[cms[i]-concentrations[i]:cms[i]])

ax.bar(x-0.5, yA, width,color=(0,0,0,0), edgecolor=colors[0]) 
ax.bar(x, yB, width,color=(0,0,0,0), edgecolor=colors[1]) 
ax.bar(x+0.5, yC, width, color=(0,0,0,0),edgecolor=colors[2]) 

ax.set_xticks(x, ['1', '3', '10','30', '90'])
ax.set_xlabel('TMT concentration ($\mu$L)')
ax.set_ylabel('Fraction of time spent in left side')
ax.legend(custom_lines, ['Phase A', 'Phase B', 'Phase C'])
ax.spines[['right', 'top']].set_visible(False)
"""