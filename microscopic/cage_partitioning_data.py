# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 14:36:58 2024

This script discretizes the cage in a mesh. Then we collect all times a mouse passes in each little square
and save the histories to file. 

@author: amaes
"""

import numpy as np
import shapely
from shapely.prepared import prep
from scipy import spatial
import matplotlib.pyplot as plt
import pickle


"""""""""""""""""""""""""""""""""
"CHOOSE POINTS IN REGULAR GRID"
"""""""""""""""""""""""""""""""""

coords = ((-1.169,14.47),(35.33,16.47),(33.915,3.4086),(50.7458,3.1977),(49.80,16.520),(85.61,13.88),
    (85.61,-21.118),( 49.36,-22.87),(50.07,-9.905),(34.75,-9.34),(35.04,-22.576),(-0.87,-19.694)) #this is static, the box does not change over time

poly = shapely.Polygon(coords)
prep_polygon = prep(poly)
valid_points = []
points = []

# Each point in mesh will be 4cm meters apart.
# You can play with this to adjust the cell size
resolution = 4

latmin, lonmin, latmax, lonmax = poly.bounds
# create mesh 
for lat in np.arange(latmin, latmax, resolution):
    for lon in np.arange(lonmin, lonmax, resolution):
        points.append(shapely.Point((round(lat,4), round(lon,4))))

# only keep those points that are within the the selected polygon.
valid_points.extend(filter(prep_polygon.contains,points))

plt.figure()
plt.plot(*poly.exterior.xy)
xs = [point.x for point in valid_points]
ys = [point.y for point in valid_points]
plt.scatter(xs, ys)

points = np.array([xs,ys]).T


""""""""""""
"LOAD DATA"
""""""""""""

model = 't_402_0'
dataset_name = 'balanced4_x2'
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


""""""""""""""""""""""""""""""""""""""""""""
"LOOP OVER POINTS AND GET DISTANCES TO DATA"
""""""""""""""""""""""""""""""""""""""""""""

history_list = []
time_offset_list = []
mice_points = states[:,0:2]
for i in range(len(points)):
    dists = np.sqrt((mice_points[:,0] - points[i,0])**2 + (mice_points[:,1] - points[i,1])**2)
    idx = np.where(dists<0.5)[0] #this is somewhat arbitrary
    idx_ = np.where( (idx[1:]-idx[0:-1])>30 )[0]+1 #again somewhat arbitrary, to cut time points that the mouse sits there
    history_list.append( states[idx[idx_],3:5] )
    
    offsets = []
    for j in range(len(idx[idx_])):
        times = idx[idx_][j] - np.hstack((np.array([0]),episode_ends))
        offset = np.where(times>0)[0]
        offsets.append(times[offset[-1]])
    time_offset_list.append(offsets)
    
    
"""""""""""""""""""""""""""
"SAVE POINTS AND HISTORIES"
""""""""""""""""""""""""""" 

with open('data/grid_points_x2.pickle', 'wb') as file:
    pickle.dump(points, file)
with open('data/grid_histories_x2.pickle', 'wb') as file:
    pickle.dump(history_list, file)
with open('data/grid_time_offsets_x2.pickle', 'wb') as file:
    pickle.dump(time_offset_list, file)   
