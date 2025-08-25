# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 11:25:37 2024

Create visualizations for the changing distribution of locations in time,
using many samples.

@author: amaes
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.stats as stats
import matplotlib.animation as animation
import shapely

model = 't_4'
curve = 'curve_0'
file = open('data_model_curves/'+model+'/'+curve+'.pickle', 'rb')
data = pickle.load(file)
file.close()

samples=data['actions']


#visualize 1D trajectory in time (only x-axis)
fig, ax = plt.subplots()

hist = ax.hist(samples[:,0,0], bins=30, histtype=u'step', density=True,color='black')
ax.set(xlabel='Position [cm]', ylabel='Density')
ax.set_xlim([-1, 85])
ax.set_ylim([0, 0.15])

def update1D(frame):
    plt.cla()
    # update the hist plot:
    hist = ax.hist(samples[:,frame,0], bins=30, histtype=u'step', density=True,color='black')
    ax.set_xlim([-1, 85])
    ax.set_ylim([0, 0.15])
    ax.set(xlabel='Position [cm]', ylabel='Density')
    return (hist)

ani = animation.FuncAnimation(fig=fig, func=update1D, frames=90, interval=10)
ani.save(filename="data_model_curves/"+model+'/'+curve+"_1Dpillow.gif", writer="pillow")



#visualize 2D trajectory in time
walls = [(-1.169,14.47),(35.33,16.47),(33.915,3.4086),(50.7458,3.1977),(49.80,16.520),(85.61,13.88),
              (85.61,-21.118),( 49.36,-22.87),(50.07,-9.905),(34.75,-9.34),(35.04,-22.576),(-0.87,-19.694)]
closed_box = walls
closed_box.append( (-1.169,14.47) ) #make sure the box is closed
closed_box = shapely.LineString( closed_box )

fig, ax = plt.subplots()

hist = ax.hist2d(samples[:,0,0],samples[:,0,1],20,cmap=plt.cm.binary)
ax.plot(*closed_box.xy,color='black')
ax.set(xlabel='Position [cm]', ylabel='Position [cm]')
ax.set_xlim([-1, 85])
ax.set_ylim([-22, 16])

def update(frame):
    # update the hist plot:
    hist = ax.hist2d(samples[:,frame,0],samples[:,frame,1],30,cmap=plt.cm.binary)
    ax.set_xlim([-1, 85])
    ax.set_ylim([-22, 16])
    return (hist)

ani = animation.FuncAnimation(fig=fig, func=update, frames=90, interval=10)
ani.save(filename="data_model_curves/"+model+'/'+curve+"_2Dpillow.gif", writer="pillow")



