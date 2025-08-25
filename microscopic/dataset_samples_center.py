# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 14:09:11 2024

See how many samples we can find in the data.

@author: amaes
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.stats as stats
import matplotlib.animation as animation
import shapely
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from matplotlib.patches import Circle

file = open('data/states_20.pickle', 'rb')
states = pickle.load(file)
file.close()

file = open('data/episode_ends_20.pickle', 'rb')
epi_ends = pickle.load(file)
file.close()

walls = [(-1.169,14.47),(35.33,16.47),(33.915,3.4086),(50.7458,3.1977),(49.80,16.520),(85.61,13.88),
              (85.61,-21.118),( 49.36,-22.87),(50.07,-9.905),(34.75,-9.34),(35.04,-22.576),(-0.87,-19.694)]
closed_box = walls
closed_box.append( (-1.169,14.47) ) #make sure the box is closed
closed_box = shapely.LineString( closed_box )

concentrations = np.array([8,10,9,9,7])

trajectories_A = []
avg_locs = []
centroid =  [18,-1] #[57,1]#[21,0] #[16,0] [16,1] [21,0] [30,-1]
offset = 12
after = 6
#PHASE A
for i in range(43):
    if i==0:
        start_idx = 0
    else:
        start_idx = epi_ends[i-1]
    end_idx = epi_ends[i]
    series = states[start_idx:end_idx,0:2]
    binary = 1*(series[:,0]>=40)
    idx = np.where((binary[1:]-binary[0:-1]!=0) & (np.abs(series[1:,1]+2)<3))[0]
            
    trajectories_A.append(states[1+start_idx+idx[0]-offset:1+start_idx+idx[0]+after,0:2])   
    avg_locs.append(states[start_idx+idx[0]+1,3:5])

    for j in range(1,np.size(idx)):
        if end_idx-start_idx-idx[j]>90 and idx[j]>idx[j-1]+90:            
            trajectories_A.append(states[1+start_idx+idx[j]-offset:1+start_idx+idx[j]+after,0:2])   
            avg_locs.append(states[start_idx+idx[j]+1,3:5])

avg_locs = np.vstack(avg_locs)
distances = np.sqrt( (avg_locs[:,0]-centroid[0])**2 + (avg_locs[:,1]-centroid[1])**2 )
idx_ = np.where(distances<5)[0]
fig, ax = plt.subplots()
for i in range(len(idx_)):
    ax.plot(trajectories_A[idx_[i]][0:offset+1,0],trajectories_A[idx_[i]][0:offset+1,1],color='green', alpha=0.2)
    ax.plot(trajectories_A[idx_[i]][offset:offset+after,0],trajectories_A[idx_[i]][offset:offset+after,1],color='grey', alpha=0.2)
    ax.scatter(trajectories_A[idx_[i]][-1,0],trajectories_A[idx_[i]][-1,1],color='grey')
ax.plot(*closed_box.xy,color='black')
ax.vlines(x=40, ymin=-5, ymax=1, color='black')
circle = Circle(tuple(centroid), 5, color='black', fill=False, edgecolor='black', linewidth=2,linestyle='--')
ax.add_patch(circle)



trajectories_B = []
avg_locs = []
tmts_B = []
colors = ['bisque','sandybrown','indianred','brown','maroon']
#PHASE B
for i in range(43,86):
    start_idx = epi_ends[i-1]
    end_idx = epi_ends[i]
    series = states[start_idx:end_idx,0:2]
    binary = 1*(series[:,0]>=40)
    idx = np.where((binary[1:]-binary[0:-1]!=0) & (np.abs(series[1:,1]+2)<3))[0]
            
    trajectories_B.append(states[1+start_idx+idx[0]-offset:1+start_idx+idx[0]+after,0:2])   
    avg_locs.append(states[start_idx+idx[0]+1,3:5])
    tmts_B.append(np.where(i-43<np.cumsum(concentrations))[0][0])
    for j in range(1,np.size(idx)):
        if end_idx-start_idx-idx[j]>90 and idx[j]>idx[j-1]+90:            
            trajectories_B.append(states[1+start_idx+idx[j]-offset:1+start_idx+idx[j]+after,0:2])   
            avg_locs.append(states[start_idx+idx[j]+1,3:5])
            tmts_B.append(np.where(i-43<np.cumsum(concentrations))[0][0])
            
avg_locs = np.vstack(avg_locs)
tmts_B = np.vstack(tmts_B).flatten()
distances = np.sqrt( (avg_locs[:,0]-centroid[0])**2 + (avg_locs[:,1]-centroid[1])**2 )
idx_ = np.where(distances<5)[0]
fig, ax = plt.subplots()
for i in range(len(idx_)):
    ax.plot(trajectories_B[idx_[i]][0:offset+1,0],trajectories_B[idx_[i]][0:offset+1,1],color='green', alpha=0.2)
    ax.plot(trajectories_B[idx_[i]][offset:offset+after,0],trajectories_B[idx_[i]][offset:offset+after,1],color=colors[tmts_B[idx_[i]]], alpha=0.2)
    ax.scatter(trajectories_B[idx_[i]][-1,0],trajectories_B[idx_[i]][-1,1],color=colors[tmts_B[idx_[i]]])
ax.plot(*closed_box.xy,color='black')
ax.vlines(x=40, ymin=-5, ymax=1, color='black')
circle = Circle(tuple(centroid), 5, color='black', fill=False, edgecolor='black', linewidth=2,linestyle='--')
ax.add_patch(circle)



trajectories_C = []
avg_locs = []
tmts_C = []
colors = ['skyblue','cornflowerblue','darkslateblue','rebeccapurple','indigo']
#PHASE C
for i in range(86,129):
    start_idx = epi_ends[i-1]
    end_idx = epi_ends[i]
    series = states[start_idx:end_idx,0:2]
    binary = 1*(series[:,0]>=40)
    idx = np.where((binary[1:]-binary[0:-1]!=0) & (np.abs(series[1:,1]+2)<3))[0]
            
    trajectories_C.append(states[1+start_idx+idx[0]-offset:1+start_idx+idx[0]+after,0:2])   
    avg_locs.append(states[start_idx+idx[0]+1,3:5])
    tmts_C.append(np.where(i-86<np.cumsum(concentrations))[0][0])
    for j in range(1,np.size(idx)):
        if end_idx-start_idx-idx[j]>90 and idx[j]>idx[j-1]+90:            
            trajectories_C.append(states[1+start_idx+idx[j]-offset:1+start_idx+idx[j]+after,0:2])   
            avg_locs.append(states[start_idx+idx[j]+1,3:5])
            tmts_C.append(np.where(i-86<np.cumsum(concentrations))[0][0])
            
avg_locs = np.vstack(avg_locs)
tmts_C = np.vstack(tmts_C).flatten()
distances = np.sqrt( (avg_locs[:,0]-centroid[0])**2 + (avg_locs[:,1]-centroid[1])**2 )
idx_ = np.where(distances<5)[0]
fig, ax = plt.subplots()
for i in range(len(idx_)):
    ax.plot(trajectories_C[idx_[i]][0:offset+1,0],trajectories_C[idx_[i]][0:offset+1,1],color='green', alpha=0.2)
    ax.plot(trajectories_C[idx_[i]][offset:offset+after,0],trajectories_C[idx_[i]][offset:offset+after,1],color=colors[tmts_C[idx_[i]]], alpha=0.2)
    ax.scatter(trajectories_C[idx_[i]][-1,0],trajectories_C[idx_[i]][-1,1],color=colors[tmts_C[idx_[i]]])
ax.plot(*closed_box.xy,color='black')
ax.vlines(x=40, ymin=-5, ymax=1, color='black')
circle = Circle(tuple(centroid), 5, color='black', fill=False, edgecolor='black', linewidth=2,linestyle='--')
ax.add_patch(circle)



# Q1: Is there memory? I mean by this: do the trajectories depend on a moving average of past location.
#     Plotting a few data trajectories, it definetely seems as this is the case.
# Q2: Can we show that models without memory display unrealistic trajectories while still modeling the distribution well?
"""
model = 't_401_1'
data_files = os.listdir('./data_model_curves/'+model+'/samples')
nb_samples = len([name for name in data_files])
samples = np.zeros((nb_samples,2,90,2)) #this is for 16 different cases, 30 seconds and x/y axis
for i in range(nb_samples):
    file = open('data_model_curves/'+model+'/samples/'+data_files[i], 'rb')
    data = pickle.load(file)
    file.close()
    samples[i,:,:,:] = data['actions']
    
    
data = np.zeros((len(idx_),90,2))
for i in range(len(idx_)):
    data[i,:,:] = trajectories[idx_[i]]
    
#visualize 1D trajectory in time (only x-axis)
fig, ax = plt.subplots()
idx_1 = 1
hist1 = ax.hist(samples[:,idx_1,0,0], bins=30, histtype=u'step', density=True,color='black')
hist2 = ax.hist(data[:,0,0], bins=30, histtype=u'step', density=True,color='green')
ax.set(xlabel='Position [cm]', ylabel='Density')
ax.set_xlim([-1, 85])
ax.set_ylim([0, 0.15])

def update1D(frame):
    plt.cla()
    # update the hist plot:
    hist1 = ax.hist(samples[:,idx_1,frame,0], bins=30, histtype=u'step', density=True,color='black')
    hist2 = ax.hist(data[:,frame,0], bins=30, histtype=u'step', density=True,color='green')
    ax.set_xlim([-1, 85])
    ax.set_ylim([0, 0.15])
    ax.set(xlabel='Position [cm]', ylabel='Density')
    return (hist1,hist2)

ani = animation.FuncAnimation(fig=fig, func=update1D, frames=90, interval=300)
ani.save(filename="data_model_curves/"+model+"/curve_baseline_R_1Dpillow.gif", writer="pillow")



#visualize 2D trajectory in time

fig, ax = plt.subplots()
div = make_axes_locatable(ax)
cax = div.append_axes('right', '5%', '5%')

ext = [[-1.5, 86], [-23, 17]]
hist1 = ax.hist2d(samples[:,idx_1,0,0],samples[:,idx_1,0,1],30,cmap=plt.cm.binary,vmin=0,vmax=8, range=ext)
hist2 = ax.hist2d(data[:,0,0],data[:,0,1],30,cmap=plt.cm.Greens,vmin=0,vmax=2, alpha=0.6, range=ext)
cb = fig.colorbar(hist1[3], cax=cax)
ax.plot(*closed_box.xy,color='black')
ax.set(xlabel='Position [cm]', ylabel='Position [cm]')
ax.set_xlim([-1, 85])
ax.set_ylim([-22, 16])

def update(frame):
    # update the hist plot:
    hist1 = ax.hist2d(samples[:,idx_1,frame,0],samples[:,idx_1,frame,1],30,cmap=plt.cm.binary,vmin=0,vmax=8, range=ext)
    hist2 = ax.hist2d(data[:,frame,0],data[:,frame,1],30,cmap=plt.cm.Greens,vmin=0,vmax=2, alpha=0.6, range=ext)
    cax.cla()
    fig.colorbar(hist1[3], cax=cax)
    ax.set_xlim([-1, 85])
    ax.set_ylim([-22, 16])
    return (hist1,hist2)

ani = animation.FuncAnimation(fig=fig, func=update, frames=90, interval=300)
ani.save(filename="data_model_curves/"+model+"/"+model+"_baseline_R_2Dpillow.gif", writer="pillow")


# fixed normalization by passing vmin=0,vmax=15 flags
"""