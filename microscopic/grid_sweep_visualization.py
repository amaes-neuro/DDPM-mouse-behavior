# -*- coding: utf-8 -*-
"""
Created on Wed Dec 4 11:42:00 2024

Visualize the grid.

@author: amaes
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import shapely
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy import stats
from astropy.stats import kuiper_two #for circular distribution!!

from dipy.segment.clustering import QuickBundles
from dipy.segment.featurespeed import (
    IdentityFeature,
)
from dipy.segment.metric import (
    AveragePointwiseEuclideanMetric,
    mdf
)

from sklearn.decomposition import PCA


plt.ioff()

model = 't_401_0'
dataset_name = 'balanced4_x1'
data_files = os.listdir('./data_model_curves/'+model+'/grid_sweep')
nb_points = len([name for name in data_files])

if not os.path.exists('figures/grid_sweep/'+model):
    os.makedirs('figures/grid_sweep/'+model)

file = open('data/grid_points_x1.pickle', 'rb')
coords = pickle.load(file)
file.close()

file = open('data/grid_histories_x1.pickle', 'rb')
hists = pickle.load(file)
file.close()

file = open('data/grid_time_offsets_x1.pickle', 'rb')
time_offsets = pickle.load(file)
file.close()

walls = [(-1.169,14.47),(35.33,16.47),(33.915,3.4086),(50.7458,3.1977),(49.80,16.520),(85.61,13.88),
              (85.61,-21.118),( 49.36,-22.87),(50.07,-9.905),(34.75,-9.34),(35.04,-22.576),(-0.87,-19.694)]
closed_box = walls
closed_box.append( (-1.169,14.47) ) #make sure the box is closed
closed_box = shapely.LineString( closed_box )

corrs_ampl = np.zeros((len(coords),))
corrs_angle = np.zeros((len(coords),))
entropy_ampl = np.zeros((len(coords),))
entropy_angle = np.zeros((len(coords),))
corrs_all_dist = []
all_data = []
data_feat = []
histories = []
offsets = []
distances = []
locs = []
for i in range(nb_points):
    grid_id = data_files[i].removesuffix('.pickle')
    grid_id = grid_id.removeprefix('curve_')
    grid_id = int(grid_id)
    file = open('data_model_curves/'+model+'/grid_sweep/'+data_files[i], 'rb')
    data = pickle.load(file)
    file.close()
    samples = data['actions'] #(nb_hist,nb_samples,16,6,2)
    if np.shape(data['state'])[0]>0:
        histories.append(hists[grid_id][data['state']])
        offsets.append(np.array(time_offsets[grid_id])[data['state']])
    smpl_avg = np.zeros((np.shape(samples)[0],16,6,2))
    
    corrs_hist = np.zeros((np.shape(samples)[0],))
    corrs_hist_ = np.zeros((np.shape(samples)[0],))
    entropy = np.zeros((np.shape(samples)[0],))
    entropy_ = np.zeros((np.shape(samples)[0],))
    for h in range(np.shape(samples)[0]):
        locs.append(coords[grid_id,:])
        distances.append(shapely.distance(shapely.Point(coords[grid_id,:]),closed_box))
        samples_sub = np.reshape(samples[h,:,:,:,:],(int(samples.shape[2]*samples.shape[1]),6,2))
        samples_sub = samples_sub - coords[grid_id,:]
        all_data.append(samples_sub)    

        delta_angle = np.zeros((np.shape(samples)[1],16,2))
        delta = np.zeros((16,2))
        total = np.zeros((16,))
        delta_end = np.zeros((16,2))
        for j in range(16):     
            total[j] = np.mean( np.sqrt( (samples[h,:,j,5,0]-coords[grid_id,0])**2 + (samples[h,:,j,5,1]-coords[grid_id,1])**2 ) )
            delta[j,0] = np.mean( np.sqrt( (samples[h,:,j,0,0]-coords[grid_id,0])**2 + (samples[h,:,j,0,1]-coords[grid_id,1])**2 ) )
            delta[j,1] = np.std(np.sqrt( (samples[h,:,j,0,0]-coords[grid_id,0])**2 + (samples[h,:,j,0,1]-coords[grid_id,1])**2 ) )
            delta_angle[:,j,0] = np.sqrt( (samples[h,:,j,0,0]-coords[grid_id,0])**2 + (samples[h,:,j,0,1]-coords[grid_id,1])**2 )
            with np.errstate(divide='raise'):
                try:
                    delta_angle[:,j,1] = np.arctan2( (samples[h,:,j,1,1]-coords[grid_id,1]),(samples[h,:,j,1,0]-coords[grid_id,0]) ) 
                except FloatingPointError:
                    delta_angle[:,j,1] = np.arctan2( (samples[h,:,j,1,1]-coords[grid_id,1]),(0.001+samples[h,:,j,1,0]-coords[grid_id,0]) )                    
                    print(i,h,j)
            smpl_avg[h,j,0,0] = np.mean(samples[h,:,j,0,0]-coords[grid_id,0] )
            smpl_avg[h,j,0,1] = np.mean(samples[h,:,j,0,1]-coords[grid_id,1] )
            for k in range(1,6):
                delta[j,0] += np.mean( np.sqrt( (samples[h,:,j,k,0]-samples[h,:,j,k-1,0])**2 + (samples[h,:,j,k,1]-samples[h,:,j,k-1,1])**2 ) )
                delta[j,1] += np.std(np.sqrt( (samples[h,:,j,k,0]-samples[h,:,j,k-1,0])**2 + (samples[h,:,j,k,1]-samples[h,:,j,k-1,1])**2 ) )
                delta_angle[:,j,0] += np.sqrt( (samples[h,:,j,k,0]-samples[h,:,j,k-1,0])**2 + (samples[h,:,j,k,1]-samples[h,:,j,k-1,1])**2 ) 
                smpl_avg[h,j,k,0] = np.mean(samples[h,:,j,k,0] -coords[grid_id,0])
                smpl_avg[h,j,k,1] = np.mean(samples[h,:,j,k,1] -coords[grid_id,1])
            
        max_delta, min_delta, tot_delta = 0, 1, 0
        max_angle, min_angle, tot_angle = 0, 1, 0
        for j in range(8):
            temp = stats.kstest(delta_angle[:,j,0],delta_angle[:,j+8,0]).statistic
            temp_ = kuiper_two(delta_angle[:,j,1],delta_angle[:,j+8,1])[0]
            tot_delta += temp
            tot_angle += temp_
            if temp > max_delta:
                max_delta = temp
            if temp < min_delta:
                min_delta = temp
            if temp_ > max_angle:
                max_angle = temp_
            if temp_ < min_angle:
                min_angle = temp_
        
        if h==5 and grid_id == 33:
            trajs = samples[h,:,3,:,:]
            trajs_ = samples[h,:,11,:,:]
            delt = delta_angle[:,3,:]
            delt_ = delta_angle[:,11,:]
        
        if np.isinf(delta_angle).any():
            print(h)
        corrs_hist[h] = np.std(delta_angle[:,8,0])/np.mean(delta_angle[:,8,0])#stats.kstest(delta_angle[:,0,0],delta_angle[:,8,0]).statistic #(max_delta-min_delta)/(max_delta+min_delta)    #np.corrcoef(delta[0:7,0], delta[7:14,0])[0,1]
        corrs_hist_[h] = np.std(delta_angle[:,14,0])/np.mean(delta_angle[:,14,0]) #kuiper_two(delta_angle[:,0,1],delta_angle[:,8,1])[0] #(max_angle-min_angle)/(max_angle+min_angle)
        entropy[h] = stats.differential_entropy(delta_angle[:,6,0]) - stats.differential_entropy(delta_angle[:,0,0]) 
        entropy_[h] = stats.differential_entropy(delta_angle[:,14,0]) - stats.differential_entropy(delta_angle[:,8,0])
    if np.shape(data['state'])[0]>0:
        corrs_ampl[grid_id] = np.mean(corrs_hist)
        corrs_angle[grid_id] = np.mean(corrs_hist_)
        entropy_ampl[grid_id] = np.mean(entropy)
        entropy_angle[grid_id] = np.mean(entropy_)
    corrs_all_dist.append(corrs_hist)

#how do i know if a correlation is significant? -> shuffle all correlations over all points 
#something like that


plt.ion()

histories = np.vstack(histories)
distances = np.vstack(distances)
offsets = np.hstack(offsets)
locs = np.vstack(locs)


corrs_ampl[corrs_ampl==0] = np.nan
fig, ax = plt.subplots(figsize=(12,4))
im=ax.scatter(coords[:,0],coords[:,1],c=corrs_ampl, marker='s', s=550, cmap='binary')
plt.colorbar(im)
ax.axis('off')
plt.savefig('figures/grid_sweep/'+model+'/'+'std_arclength_food_0.pdf', format="pdf", bbox_inches="tight")

corrs_angle[corrs_angle==0] = np.nan
fig, ax = plt.subplots(figsize=(12,4))
im=ax.scatter(coords[:,0],coords[:,1],c=corrs_angle, marker='s', s=550, cmap='binary')
plt.colorbar(im)
ax.axis('off')
plt.savefig('figures/grid_sweep/'+model+'/'+'std_arclength_food_6.pdf', format="pdf", bbox_inches="tight")


entropy_ampl[entropy_ampl==0] = np.nan
fig, ax = plt.subplots(figsize=(12,4))
im=ax.scatter(coords[:,0],coords[:,1],c=entropy_ampl, marker='s', s=550, cmap='bwr',vmin=-1.5,vmax=1.5)
plt.colorbar(im)
ax.axis('off')
plt.savefig('figures/grid_sweep/'+model+'/'+'entropy_arclength_tmt.pdf', format="pdf", bbox_inches="tight")

entropy_angle[entropy_angle==0] = np.nan
fig, ax = plt.subplots(figsize=(12,4))
im=ax.scatter(coords[:,0],coords[:,1],c=entropy_angle, marker='s', s=550, cmap='bwr',vmin=-1.5,vmax=1.5)
plt.colorbar(im)
ax.axis('off')
plt.savefig('figures/grid_sweep/'+model+'/'+'entropy_arclength_food.pdf', format="pdf", bbox_inches="tight")


"""

if os.path.exists('./data_model_curves/'+model+'/feats_'+model+'.pickle'):
    file = open('data_model_curves/'+model+'/feats_'+model+'.pickle', 'rb')
    feats = pickle.load(file)
    file.close()
else:
    feats = np.zeros((len(all_data),int((1600**2-1600)/2)))
    for i in range(len(all_data)):
        it = 0
        print('Iteration '+str(i))
        for j in range(1600-1):
            for k in range(j+1,1600):
                feats[i,it] = mdf(all_data[i][j,:,:],all_data[i][k,:,:])
                it += 1
    with open('data_model_curves/'+model+'/feats_'+model+'.pickle', 'wb') as file:
        pickle.dump(feats, file)


bins = np.arange(0,np.max(feats)+np.max(feats)/100,np.max(feats)/100)
feats_transform = np.zeros((feats.shape[0],bins.shape[0]-1))
for i in range(feats.shape[0]):
    n,bins = np.histogram(feats[i,:],bins)
    feats_transform[i,:] = n

#pca = PCA(n_components=100)
#feats_transform = pca.fit_transform(feats)
X_embedded = TSNE(n_components=2, learning_rate='auto',
                  init='random', perplexity=30).fit_transform(feats_transform)

nb_clust = 10
kmeans = KMeans(n_clusters=nb_clust)
kmeans.fit(X_embedded)

fig, ax = plt.subplots()
ax.scatter(X_embedded[:,0], X_embedded[:,1],c='grey', alpha=.2);
ax.axis('off')
plt.savefig('figures/grid_sweep/'+model+'/'+'embedding.pdf', format="pdf", bbox_inches="tight")

fig, ax = plt.subplots()
ax.bar(bins[0:-1],feats_transform[469],color='grey')
plt.savefig('figures/grid_sweep/'+model+'/'+'embedding_469.pdf', format="pdf", bbox_inches="tight")

fig, ax = plt.subplots()
ax.bar(bins[0:-1],feats_transform[6],color='grey')
plt.savefig('figures/grid_sweep/'+model+'/'+'embedding_6.pdf', format="pdf", bbox_inches="tight")

fig, ax = plt.subplots()
ax.bar(bins[0:-1],feats_transform[181],color='grey')
plt.savefig('figures/grid_sweep/'+model+'/'+'embedding_181.pdf', format="pdf", bbox_inches="tight")

fig, ax = plt.subplots()
ax.bar(bins[0:-1],feats_transform[248],color='grey')
plt.savefig('figures/grid_sweep/'+model+'/'+'embedding_248.pdf', format="pdf", bbox_inches="tight")


#i can screen this point cloud for: distance to walls, location history, distance to food, time offset since trial start,...
#instead of pca on pairwise distances, can keep bins of histogram
#this also seems to do the trick, are there other interesting things to look at in this low-d space?
#I could also look at individual sets of 100 trajectories instead of pooling all food no food tmt sets in one point
fig, ax = plt.subplots()
sc = ax.scatter(X_embedded[:,0], X_embedded[:,1],c=distances,alpha=0.5, cmap=plt.cm.RdYlGn);
plt.colorbar(sc)
ax.axis('off')
plt.savefig('figures/grid_sweep/'+model+'/'+'embedding_distance_walls.pdf', format="pdf", bbox_inches="tight")

fig, ax = plt.subplots()
sc = ax.scatter(X_embedded[:,0], X_embedded[:,1],c=np.sqrt(locs[:,0]**2+locs[:,1]**2),alpha=0.5, cmap=plt.cm.RdYlGn);
plt.colorbar(sc)
ax.axis('off')
plt.savefig('figures/grid_sweep/'+model+'/'+'embedding_distance_food.pdf', format="pdf", bbox_inches="tight")

shift = locs-histories 
fig, ax = plt.subplots()
sc = ax.scatter(X_embedded[:,0], X_embedded[:,1],c=shift[:,0],alpha=0.5, cmap=plt.cm.RdYlGn);
plt.colorbar(sc)
ax.axis('off')
plt.savefig('figures/grid_sweep/'+model+'/'+'embedding_history.pdf', format="pdf", bbox_inches="tight")

#time offsets in minutes
fig, ax = plt.subplots()
sc = ax.scatter(X_embedded[:,0], X_embedded[:,1],c=offsets/180,alpha=0.5, cmap=plt.cm.RdYlGn);
plt.colorbar(sc)
ax.axis('off')
plt.savefig('figures/grid_sweep/'+model+'/'+'embedding_time.pdf', format="pdf", bbox_inches="tight")


feature = IdentityFeature()
metric = AveragePointwiseEuclideanMetric(feature=feature)
qb = QuickBundles(threshold=10, metric=metric)

idx = []
nb_rows = 5
fig, ax = plt.subplots(nb_rows,nb_clust)
for i in range(nb_clust):
    idx.append( np.where(kmeans.labels_==i)[0] )
    
    for k in range(nb_rows):
        idx_ = idx[i][5*k]
        cluster = qb.cluster(all_data[idx_])
        for j in range(len(cluster)):
            ax[k,i].plot(cluster.centroids[j][:,0],cluster.centroids[j][:,1],c='b')
plt.savefig('figures/grid_sweep/'+model+'/'+'trajectory_qb_clusters.pdf', format="pdf", bbox_inches="tight")




"""








"""
feature = IdentityFeature()
metric = AveragePointwiseEuclideanMetric(feature=feature)
qb = QuickBundles(threshold=10, metric=metric)

clusters = []
nb_clust = 15
feats = np.zeros((len(all_data),int((nb_clust**2-nb_clust)/2)))
for i in range(len(all_data)):
    print('Iteration '+str(i))
    threshold = 5
    cluster = qb.cluster(all_data[i])
    change = 1
    it = 1
    while len(cluster) != nb_clust:
        if len(cluster)>nb_clust:
            threshold += change/it
            qb = QuickBundles(threshold=threshold, metric=metric)
            cluster = qb.cluster(all_data[i])
        else:
            threshold -= change/it
            qb = QuickBundles(threshold=threshold, metric=metric)
            cluster = qb.cluster(all_data[i])
        it += 1*(it<=6000)
        #if i==128:
        #    print(len(cluster))
        if it>6000 and len(cluster) == nb_clust + 1:
            break
    clusters.append(cluster)
    it = 0
    for j in range(nb_clust-1):
        for k in range(j+1,nb_clust):
            feats[i,it] = mdf(cluster.centroids[j],cluster.centroids[k])
            it += 1
"""


"""
" APPLY TSNE DIMENSIONALITY REDUCTION "
"""

"""
#concatenate the 14 2sec average trajectories to one feature vector (168D), for each location and 10 histories (1592 vectors or so)
#then do tSNE and identify if indeed there are distinct clusters. 
#Intuitively we would expect tmt-sensitive cluster, food-sensitive, mixed,...

all_data = np.vstack(all_data)
X = all_data[:,:,0:4,:]
X = X.reshape((X.shape[0],112))
X_embedded = TSNE(n_components=2, learning_rate='auto',
                  init='random', perplexity=30).fit_transform(X)

nb_clust = 4
kmeans = KMeans(n_clusters=nb_clust)
kmeans.fit(X_embedded)

fig, ax = plt.subplots()
ax.scatter(X_embedded[:,0], X_embedded[:,1],c=kmeans.labels_, alpha=.1);

#how do the trajectories look like?
idx = []
for i in range(nb_clust):
    idx.append( np.where(kmeans.labels_==i)[0] )
    
    idx_ = idx[i][10]
    fig, ax = plt.subplots()
    for j in range(14):
        c = (0,0,1,(j+1)/7)
        if j>=7:
            c = (1,0,0,(j-6)/7)
        ax.plot(all_data[idx_,j,:,0],all_data[idx_,j,:,1],c=c)
        ax.set_xlabel('x-position [cm]')
        ax.set_ylabel('y-position [cm]')

    
idx = []
nb_rows = 7
fig, ax = plt.subplots(nb_rows,nb_clust)
for i in range(nb_clust):
    idx.append( np.where(kmeans.labels_==i)[0] )
    
    for k in range(nb_rows):
        idx_ = idx[i][5*k]
    
        for j in range(14):
            c = (0,0,1,(j+1)/7)
            if j>=7:
                c = (1,0,0,(j-6)/7)
            ax[k,i].plot(all_data[idx_,j,:,0],all_data[idx_,j,:,1],c=c)
    
plt.savefig('figures/grid_sweep/'+model+'/'+'trajectory_clusters.pdf', format="pdf", bbox_inches="tight")
plt.close()
"""