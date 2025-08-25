# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 12:57:40 2024

Combine samples into one variable and make visualizations.

@author: amaes
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.stats as stats
import matplotlib.animation as animation
import shapely
from mpl_toolkits.axes_grid1 import make_axes_locatable

#it is possible some samples were not computed properly because of a bad GPU, this might affect up to 1% roughly :/
model = 't_4'
nb_samples = 1000
samples = np.zeros((nb_samples,16,90,2)) #this is for 16 different cases, 30 seconds and x/y axis
for i in range(nb_samples):
    file = open('data_model_curves/'+model+'/samples/curve_'+str(i)+'.pickle', 'rb')
    data = pickle.load(file)
    file.close()
    samples[i,:,:,:] = data['actions']
    
    
#visualize 1D trajectory in time (only x-axis)
fig, ax = plt.subplots()
idx_1 = 0
idx_2 = 4
hist1 = ax.hist(samples[:,idx_1,0,0], bins=30, histtype=u'step', density=True,color='black')
hist2 = ax.hist(samples[:,idx_2,0,0], bins=30, histtype=u'step', density=True,color='green')
ax.set(xlabel='Position [cm]', ylabel='Density')
ax.set_xlim([-1, 85])
ax.set_ylim([0, 0.15])

def update1D(frame):
    plt.cla()
    # update the hist plot:
    hist1 = ax.hist(samples[:,idx_1,frame,0], bins=30, histtype=u'step', density=True,color='black')
    hist2 = ax.hist(samples[:,idx_2,frame,0], bins=30, histtype=u'step', density=True,color='green')
    ax.set_xlim([-1, 85])
    ax.set_ylim([0, 0.15])
    ax.set(xlabel='Position [cm]', ylabel='Density')
    return (hist1,hist2)

ani = animation.FuncAnimation(fig=fig, func=update1D, frames=90, interval=200)
ani.save(filename="data_model_curves/"+model+'/curve_'+str(idx_1)+'_'+str(idx_2)+"_1Dpillow.gif", writer="pillow")


#visualize 2D trajectory in time
walls = [(-1.169,14.47),(35.33,16.47),(33.915,3.4086),(50.7458,3.1977),(49.80,16.520),(85.61,13.88),
              (85.61,-21.118),( 49.36,-22.87),(50.07,-9.905),(34.75,-9.34),(35.04,-22.576),(-0.87,-19.694)]
closed_box = walls
closed_box.append( (-1.169,14.47) ) #make sure the box is closed
closed_box = shapely.LineString( closed_box )

fig, ax = plt.subplots()
div = make_axes_locatable(ax)
cax = div.append_axes('right', '5%', '5%')

ext = [[-1, 85], [-22, 16]]
hist1 = ax.hist2d(samples[:,idx_1,0,0],samples[:,idx_1,0,1],30,cmap=plt.cm.binary,vmin=0,vmax=15, range=ext)
hist2 = ax.hist2d(samples[:,idx_2,0,0],samples[:,idx_2,0,1],30,cmap=plt.cm.Greens,vmin=0,vmax=15, alpha=0.6, range=ext)
cb = fig.colorbar(hist1[3], cax=cax)
ax.plot(*closed_box.xy,color='black')
ax.set(xlabel='Position [cm]', ylabel='Position [cm]')
ax.set_xlim([-1, 85])
ax.set_ylim([-22, 16])

def update(frame):
    # update the hist plot:
    hist1 = ax.hist2d(samples[:,idx_1,frame,0],samples[:,idx_1,frame,1],30,cmap=plt.cm.binary,vmin=0,vmax=15, range=ext)
    hist2 = ax.hist2d(samples[:,idx_2,frame,0],samples[:,idx_2,frame,1],30,cmap=plt.cm.Greens,vmin=0,vmax=15, alpha=0.6, range=ext)
    cax.cla()
    fig.colorbar(hist1[3], cax=cax)
    ax.set_xlim([-1, 85])
    ax.set_ylim([-22, 16])
    return (hist1,hist2)

ani = animation.FuncAnimation(fig=fig, func=update, frames=90, interval=200)
ani.save(filename="data_model_curves/"+model+'/curve_'+str(idx_1)+'_'+str(idx_2)+"_2Dpillow.gif", writer="pillow")

# fixed normalization by passing vmin, vmax flags to colorbar
#TODO add some trajectories from the data to indicate that the distribution might be 'centered' around this
#I implemented this in a new script.

#TODO compare samples from balanced2 model with balanced4, how do the trajectories change when adding a memory?
#Intuition: they should be more 'directed'



