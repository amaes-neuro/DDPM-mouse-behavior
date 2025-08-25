# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 09:58:00 2025

Analyse food approaches and retreats.

Do this both for food present or not present. 

We use quickbundles in the dipy package to cluster trajectories.

@author: amaes
"""

import numpy as np

from dipy.io.pickles import save_pickle
from dipy.segment.clustering import QuickBundles
from dipy.segment.featurespeed import (
    ArcLengthFeature,
    CenterOfMassFeature,
    IdentityFeature,
    MidpointFeature,
    ResampleFeature,
    VectorOfEndpointsFeature,
)
from dipy.segment.metric import (
    AveragePointwiseEuclideanMetric,
    CosineMetric,
    EuclideanMetric,
)

import matplotlib.pyplot as plt
import pickle
import os
import shapely


model = 't_401_0'
dataset_name = 'balanced4_x1'
data_files = os.listdir('./data_model_curves/'+model+'/grid_sweep')
nb_points = len([name for name in data_files])

if not os.path.exists('figures/grid_sweep/'+model):
    os.makedirs('figures/grid_sweep/'+model)

file = open('data/grid_points_x1.pickle', 'rb')
coords = pickle.load(file)
file.close()

x_, y_ = 30, 0
idx_approach = np.where( np.sqrt((coords[:,0]-x_)**2 + (coords[:,1]-y_)**2)<5)[0] #can change this of course
idx_retreat = np.where( np.sqrt(coords[:,0]**2 + coords[:,1]**2)<5)[0]

approaches, approaches_labels = [], []
retreats, retreats_labels = [] ,[]
for i in range(nb_points):
    grid_id = data_files[i].removesuffix('.pickle')
    grid_id = grid_id.removeprefix('curve_')
    grid_id = int(grid_id)
    file = open('data_model_curves/'+model+'/grid_sweep/'+data_files[i], 'rb')
    data = pickle.load(file)
    file.close()
    samples = data['actions'] #(nb_hist,nb_samples,16,6,2)
            
    if grid_id in idx_approach:
        for j in range(samples.shape[2]):
            samples_sub = samples[:,:,j,:,:]
            samples_sub = np.reshape(samples_sub,(int(samples.shape[0]*samples.shape[1]),6,2))
            idx_appr = np.where(samples_sub[:,5,0]+2<coords[grid_id,0])[0]
            approaches.append(samples_sub[idx_appr,:,:]-coords[grid_id,:])
            approaches_labels.append(j*np.ones((len(idx_appr),1)))
    elif grid_id in idx_retreat:
        for j in range(samples.shape[2]):
            samples_sub = samples[:,:,j,:,:]
            samples_sub = np.reshape(samples_sub,(int(samples.shape[0]*samples.shape[1]),6,2))
            idx_retr = np.where(samples_sub[:,5,0]-2>coords[grid_id,0])[0]
            retreats.append(samples_sub[idx_retr,:,:]-coords[grid_id,:])
            retreats_labels.append(j*np.ones((len(idx_retr),1)))
       
approaches = np.vstack(approaches)
approaches_labels = np.vstack(approaches_labels)
retreats = np.vstack(retreats)
retreats_labels = np.vstack(retreats_labels)
    
idx_appr_food = np.where(approaches_labels>=8)[0]
idx_appr_tmt = np.where(approaches_labels<8)[0]
approaches_food = approaches[idx_appr_food,:,:]
approaches_food_labels = approaches_labels[idx_appr_food]
approaches_tmt = approaches[idx_appr_tmt,:,:]    
approaches_tmt_labels = approaches_labels[idx_appr_tmt]

idx_retr_food = np.where(retreats_labels>=8)[0]
idx_retr_tmt = np.where(retreats_labels<8)[0]
retreats_food = retreats[idx_retr_food,:,:]
retreats_food_labels = retreats_labels[idx_retr_food]
retreats_tmt = retreats[idx_retr_tmt,:,:]    
retreats_tmt_labels = retreats_labels[idx_retr_tmt]

feature = IdentityFeature()
metric = AveragePointwiseEuclideanMetric(feature=feature)
qb1 = QuickBundles(threshold=10, metric=metric)
qb2 = QuickBundles(threshold=13, metric=metric)

clusters_appr_tmt = qb1.cluster(approaches_tmt)
clusters_appr_food = qb1.cluster(approaches_food)
clusters_retr_tmt = qb2.cluster(retreats_tmt)
clusters_retr_food = qb2.cluster(retreats_food)


walls = [(-1.169,14.47),(35.33,16.47),(33.915,3.4086),(50.7458,3.1977),(49.80,16.520),(85.61,13.88),
              (85.61,-21.118),( 49.36,-22.87),(50.07,-9.905),(34.75,-9.34),(35.04,-22.576),(-0.87,-19.694)]
closed_box = walls
closed_box.append( (-1.169,14.47) ) #make sure the box is closed
closed_box = shapely.LineString( closed_box )

appr_angles_food = np.zeros((len(clusters_appr_food.centroids),3))
appr_angles_tmt = np.zeros((len(clusters_appr_tmt.centroids),3))
fig, ax = plt.subplots()
for i in range(len(clusters_appr_food.centroids)):
    appr_angles_food[i,0] = np.mean(approaches_food_labels[clusters_appr_food[i].indices]-8)
    appr_angles_food[i,1] = np.arctan2(clusters_appr_food.centroids[i][2,1],clusters_appr_food.centroids[i][2,0])
    appr_angles_food[i,2] = np.min(np.sqrt( (x_+clusters_appr_food.centroids[i][:,0])**2 +  (y_+clusters_appr_food.centroids[i][:,1])**2 ) )
    alpha = np.mean(approaches_food_labels[clusters_appr_food[i].indices]-8)/8
    ax.plot(x_+clusters_appr_food.centroids[i][:,0],y_+clusters_appr_food.centroids[i][:,1],c='yellowgreen',alpha=alpha)

for i in range(len(clusters_appr_tmt.centroids)):
    appr_angles_tmt[i,0] = np.mean(approaches_tmt_labels[clusters_appr_tmt[i].indices])
    appr_angles_tmt[i,1] = np.arctan2(clusters_appr_tmt.centroids[i][2,1],clusters_appr_tmt.centroids[i][2,0])
    appr_angles_tmt[i,2] = np.min(np.sqrt( (x_+clusters_appr_tmt.centroids[i][:,0])**2 + (y_+clusters_appr_tmt.centroids[i][:,1])**2 ))
    alpha = np.mean(approaches_tmt_labels[clusters_appr_tmt[i].indices])/8
    ax.plot(x_+clusters_appr_tmt.centroids[i][:,0],y_+clusters_appr_tmt.centroids[i][:,1],c='#006400',alpha=alpha)
ax.plot(*closed_box.xy,color='black')
ax.axis('off')
plt.savefig('figures/grid_sweep/'+model+'/'+'approach.pdf', format="pdf", bbox_inches="tight")


retr_angles_food = np.zeros((len(clusters_retr_food.centroids),3))
retr_angles_tmt = np.zeros((len(clusters_retr_tmt.centroids),3))
fig, ax = plt.subplots()
for i in range(len(clusters_retr_food.centroids)):
    retr_angles_food[i,0] = np.mean(retreats_food_labels[clusters_retr_food[i].indices]-8)
    retr_angles_food[i,1] = np.arctan2(clusters_retr_food.centroids[i][2,1],clusters_retr_food.centroids[i][2,0])
    retr_angles_food[i,2] = np.max(np.sqrt(clusters_retr_food.centroids[i][:,0]**2+clusters_retr_food.centroids[i][:,1]**2))
    alpha = np.mean(retreats_food_labels[clusters_retr_food[i].indices]-8)/8
    ax.plot(clusters_retr_food.centroids[i][:,0],clusters_retr_food.centroids[i][:,1],c='yellowgreen',alpha=alpha)

for i in range(len(clusters_retr_tmt.centroids)):
    retr_angles_tmt[i,0] = np.mean(retreats_tmt_labels[clusters_retr_tmt[i].indices])
    retr_angles_tmt[i,1] = np.arctan2(clusters_retr_tmt.centroids[i][2,1],clusters_retr_tmt.centroids[i][2,0])
    retr_angles_tmt[i,2] = np.max(np.sqrt(clusters_retr_tmt.centroids[i][:,0]**2 + clusters_retr_tmt.centroids[i][:,1]**2))
    alpha = np.mean(retreats_tmt_labels[clusters_retr_tmt[i].indices])/8
    ax.plot(clusters_retr_tmt.centroids[i][:,0],clusters_retr_tmt.centroids[i][:,1],c='#006400',alpha=alpha)
ax.plot(*closed_box.xy,color='black')
ax.axis('off')
plt.savefig('figures/grid_sweep/'+model+'/'+'retreat.pdf', format="pdf", bbox_inches="tight")


fig,ax = plt.subplots()
ax.scatter(appr_angles_food[:,1],appr_angles_food[:,0]/10,c='yellowgreen')
ax.scatter(appr_angles_tmt[:,1],appr_angles_tmt[:,0]/10,c='#006400')
ax.scatter(retr_angles_food[:,1],retr_angles_food[:,0]/10,c='yellowgreen',marker='s')
ax.scatter(retr_angles_tmt[:,1],retr_angles_tmt[:,0]/10,c='#006400',marker='s')
ax.spines[['right', 'top']].set_visible(False)
ax.set_xlabel('Angle of approach/retreat [rad]')
ax.set_ylabel('Average TMT level in cluster')
ax.set_ylim(0, 0.75)
ax.set_xlim(-3.15,3.15)
plt.savefig('figures/grid_sweep/'+model+'/'+'angles_approach_retreat.pdf', format="pdf", bbox_inches="tight")

fig,ax = plt.subplots()
ax.scatter(appr_angles_food[:,2],appr_angles_food[:,0]/10,c='yellowgreen')
ax.scatter(appr_angles_tmt[:,2],appr_angles_tmt[:,0]/10,c='#006400')
ax.scatter(retr_angles_food[:,2],retr_angles_food[:,0]/10,c='yellowgreen',marker='s')
ax.scatter(retr_angles_tmt[:,2],retr_angles_tmt[:,0]/10,c='#006400',marker='s')
ax.spines[['right', 'top']].set_visible(False)
ax.set_ylim(0, 0.75)
ax.set_xlim(0,85)
ax.set_xlabel('Distance to food location [cm]')
ax.set_ylabel('Average TMT level in cluster')
plt.savefig('figures/grid_sweep/'+model+'/'+'distances_approach_retreat.pdf', format="pdf", bbox_inches="tight")





"""
RETREATS AND APPROACHES IN DATASET
"""

dataset = dataset_name

file = open('data/states_'+dataset+'.pickle', 'rb')
states = pickle.load(file)
file.close()

file = open('data/episode_ends_'+dataset+'.pickle', 'rb')
epi_ends = pickle.load(file)
file.close()

data_appr_food = []
data_appr_tmt = []
data_appr_food_labels = []
data_appr_tmt_labels = []
data_retr_food = []
data_retr_tmt = []
data_retr_food_labels = []
data_retr_tmt_labels = []
for i in range(0,93):
    if i==0:
        start_idx = 0
    else:
        start_idx = epi_ends[i-1]
    end_idx = epi_ends[i]
    series = states[start_idx:end_idx,0:2]
    lbl = states[start_idx,7]
    j = 0
    while j < series.shape[0]-7:
        if np.sqrt(series[j,0]**2 + series[j,1]**2) < 5 and (series[j+6,0]>series[j,0]+2):
            if states[start_idx,6] == 1:
                data_retr_food.append(series[j+1:j+7,:]-series[j,:])
                data_retr_food_labels.append(lbl)
            else:
                data_retr_tmt.append(series[j+1:j+7,:]-series[j,:])
                data_retr_tmt_labels.append(lbl)
            j += 7
        elif np.sqrt((series[j,0]-x_)**2 + (series[j,1]-y_)**2) < 5 and (series[j+6,0]<series[j,0]-2):
            if states[start_idx,6] == 1:
                data_appr_food.append(series[j+1:j+7,:]-series[j,:])
                data_appr_food_labels.append(lbl)
            else:
                data_appr_tmt.append(series[j+1:j+7,:]-series[j,:])
                data_appr_tmt_labels.append(lbl)
            j += 7
        j += 1

data_retr_food_labels = np.vstack(data_retr_food_labels)
data_retr_tmt_labels = np.vstack(data_retr_tmt_labels)
data_appr_food_labels = np.vstack(data_appr_food_labels)
data_appr_tmt_labels = np.vstack(data_appr_tmt_labels)

qb1 = QuickBundles(threshold=8, metric=metric)
qb2 = QuickBundles(threshold=11, metric=metric)

clusters_data_appr_tmt = qb1.cluster(data_appr_tmt)
clusters_data_appr_food = qb1.cluster(data_appr_food)
clusters_data_retr_tmt = qb2.cluster(data_retr_tmt)
clusters_data_retr_food = qb2.cluster(data_retr_food)

data_appr_angles_food = np.zeros((len(clusters_data_appr_food.centroids),3))
data_appr_angles_tmt = np.zeros((len(clusters_data_appr_tmt.centroids),3))
fig, ax = plt.subplots()
for i in range(len(clusters_data_appr_food.centroids)):
    data_appr_angles_food[i,0] = np.mean(data_appr_food_labels[clusters_data_appr_food[i].indices])
    data_appr_angles_food[i,1] = np.arctan2(clusters_data_appr_food.centroids[i][2,1],clusters_data_appr_food.centroids[i][2,0])
    data_appr_angles_food[i,2] = np.min(np.sqrt( (x_+clusters_data_appr_food.centroids[i][:,0])**2 +  (y_+clusters_data_appr_food.centroids[i][:,1])**2 ) )
    alpha = np.mean(data_appr_food_labels[clusters_data_appr_food[i].indices])
    ax.plot(x_+clusters_data_appr_food.centroids[i][:,0],y_+clusters_data_appr_food.centroids[i][:,1],c='yellowgreen',alpha=alpha)

for i in range(len(clusters_data_appr_tmt.centroids)):
    data_appr_angles_tmt[i,0] = np.mean(data_appr_tmt_labels[clusters_data_appr_tmt[i].indices])
    data_appr_angles_tmt[i,1] = np.arctan2(clusters_data_appr_tmt.centroids[i][2,1],clusters_data_appr_tmt.centroids[i][2,0])
    data_appr_angles_tmt[i,2] = np.min(np.sqrt( (x_+clusters_data_appr_tmt.centroids[i][:,0])**2 + (y_+clusters_data_appr_tmt.centroids[i][:,1])**2 ))
    alpha = np.mean(data_appr_tmt_labels[clusters_data_appr_tmt[i].indices])
    ax.plot(x_+clusters_data_appr_tmt.centroids[i][:,0],y_+clusters_data_appr_tmt.centroids[i][:,1],c='#006400',alpha=alpha)
ax.plot(*closed_box.xy,color='black')
ax.axis('off')
plt.savefig('figures/grid_sweep/'+model+'/'+'approach_DATA.pdf', format="pdf", bbox_inches="tight")


data_retr_angles_food = np.zeros((len(clusters_data_retr_food.centroids),3))
data_retr_angles_tmt = np.zeros((len(clusters_data_retr_tmt.centroids),3))
fig, ax = plt.subplots()
for i in range(len(clusters_data_retr_food.centroids)):
    data_retr_angles_food[i,0] = np.mean(data_retr_food_labels[clusters_data_retr_food[i].indices])
    data_retr_angles_food[i,1] = np.arctan2(clusters_data_retr_food.centroids[i][2,1],clusters_data_retr_food.centroids[i][2,0])
    data_retr_angles_food[i,2] = np.max(np.sqrt(clusters_data_retr_food.centroids[i][5:,0]**2+clusters_data_retr_food.centroids[i][:,1]**2))
    alpha = np.mean(data_retr_food_labels[clusters_data_retr_food[i].indices])
    ax.plot(clusters_data_retr_food.centroids[i][:,0],clusters_data_retr_food.centroids[i][:,1],c='yellowgreen',alpha=alpha)

for i in range(len(clusters_data_retr_tmt.centroids)):
    data_retr_angles_tmt[i,0] = np.mean(data_retr_tmt_labels[clusters_data_retr_tmt[i].indices])
    data_retr_angles_tmt[i,1] = np.arctan2(clusters_data_retr_tmt.centroids[i][2,1],clusters_data_retr_tmt.centroids[i][2,0])
    data_retr_angles_tmt[i,2] = np.max(np.sqrt(clusters_data_retr_tmt.centroids[i][:,0]**2 + clusters_data_retr_tmt.centroids[i][:,1]**2))
    alpha = np.mean(data_retr_tmt_labels[clusters_data_retr_tmt[i].indices])
    ax.plot(clusters_data_retr_tmt.centroids[i][:,0],clusters_data_retr_tmt.centroids[i][:,1],c='#006400',alpha=alpha)
ax.plot(*closed_box.xy,color='black')
ax.axis('off')
plt.savefig('figures/grid_sweep/'+model+'/'+'retreat_DATA.pdf', format="pdf", bbox_inches="tight")

fig,ax = plt.subplots()
ax.scatter(data_appr_angles_food[:,1],data_appr_angles_food[:,0],c='yellowgreen')
ax.scatter(data_appr_angles_tmt[:,1],data_appr_angles_tmt[:,0],c='#006400')
ax.scatter(data_retr_angles_food[:,1],data_retr_angles_food[:,0],c='yellowgreen',marker='s')
ax.scatter(data_retr_angles_tmt[:,1],data_retr_angles_tmt[:,0],c='#006400',marker='s')
ax.spines[['right', 'top']].set_visible(False)
ax.set_ylim(0, 0.75)
ax.set_xlim(-3.15,3.15)
ax.set_xlabel('Angle of approach/retreat [rad]')
ax.set_ylabel('Average TMT level in cluster')
plt.savefig('figures/grid_sweep/'+model+'/'+'angles_approach_retreat_DATA.pdf', format="pdf", bbox_inches="tight")

fig,ax = plt.subplots()
ax.scatter(data_appr_angles_food[:,2],data_appr_angles_food[:,0],c='yellowgreen')
ax.scatter(data_appr_angles_tmt[:,2],data_appr_angles_tmt[:,0],c='#006400')
ax.scatter(data_retr_angles_food[:,2],data_retr_angles_food[:,0],c='yellowgreen',marker='s')
ax.scatter(data_retr_angles_tmt[:,2],data_retr_angles_tmt[:,0],c='#006400',marker='s')
ax.spines[['right', 'top']].set_visible(False)
ax.set_ylim(0, 0.75)
ax.set_xlim(0, 85)
ax.set_xlabel('Distance to food location [cm]')
ax.set_ylabel('Average TMT level in cluster')
plt.savefig('figures/grid_sweep/'+model+'/'+'distances_approach_retreat_DATA.pdf', format="pdf", bbox_inches="tight")


