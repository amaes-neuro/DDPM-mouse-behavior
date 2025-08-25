# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:50:15 2025

Visualize the grid, only the effect of TMT when food is not present.

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
from astropy.stats import kuiper_two

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

corrs_ampl = np.zeros((len(coords),))
corrs_ampl_std = np.zeros((len(coords),))
corrs_angle = np.zeros((len(coords),))
corrs_angle_std = np.zeros((len(coords),))
corrs_all_dist = []
all_data = []
data_feat = []
for i in range(nb_points):
    grid_id = data_files[i].removesuffix('.pickle')
    grid_id = grid_id.removeprefix('curve_')
    grid_id = int(grid_id)
    file = open('data_model_curves/'+model+'/grid_sweep/'+data_files[i], 'rb')
    data = pickle.load(file)
    file.close()
    samples = data['actions'] #(nb_hist,nb_samples,14,6,2)
    smpl_avg = np.zeros((np.shape(samples)[0],8,6,2))
    
    corrs_hist = np.zeros((np.shape(samples)[0],))
    corrs_hist_ = np.zeros((np.shape(samples)[0],))
    for h in range(np.shape(samples)[0]):
        delta_angle = np.zeros((np.shape(samples)[1],8,2))
        delta = np.zeros((8,2))
        total = np.zeros((8,))
        delta_end = np.zeros((8,2))
        for j in range(8):     
            total[j] = np.mean( np.sqrt( (samples[h,:,j,5,0]-coords[grid_id,0])**2 + (samples[h,:,j,5,1]-coords[grid_id,1])**2 ) )
            delta[j,0] = np.mean( np.sqrt( (samples[h,:,j,0,0]-coords[grid_id,0])**2 + (samples[h,:,j,0,1]-coords[grid_id,1])**2 ) )
            delta[j,1] = np.std(np.sqrt( (samples[h,:,j,0,0]-coords[grid_id,0])**2 + (samples[h,:,j,0,1]-coords[grid_id,1])**2 ) )
            delta_angle[:,j,0] = np.sqrt( (samples[h,:,j,0,0]-coords[grid_id,0])**2 + (samples[h,:,j,0,1]-coords[grid_id,1])**2 )
            delta_angle[:,j,1] = np.arctan2( (samples[h,:,j,1,1]-coords[grid_id,1]),(samples[h,:,j,1,0]-coords[grid_id,0]) )
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
        for j in range(7):
            temp = stats.kstest(delta_angle[:,j,0],delta_angle[:,j+1,0]).statistic
            temp_ = kuiper_two(delta_angle[:,j,1],delta_angle[:,j+1,1])[0]
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

        corrs_hist[h] =  stats.kstest(delta_angle[:,1,0],delta_angle[:,6,0]).statistic #max_delta/tot_delta#(max_delta-min_delta)/(max_delta+min_delta)    #(np.max(delta[:,0])-np.min(delta[:,0]))/np.max(delta[:,0])
        corrs_hist_[h] = kuiper_two(delta_angle[:,1,1],delta_angle[:,6,1])[0] #(max_angle-min_angle)/(max_angle+min_angle)   #np.arccos( np.dot(smpl_avg[h,0,0,:],smpl_avg[h,4,0,:])/(np.linalg.norm(smpl_avg[h,0,0,:])*np.linalg.norm(smpl_avg[h,4,0,:])) )
    corrs_ampl[grid_id] = np.mean(corrs_hist)
    corrs_ampl_std[grid_id] = np.std(corrs_hist)
    corrs_angle[grid_id] = np.mean(corrs_hist_)
    corrs_angle_std[grid_id] = np.mean(corrs_hist_)
    corrs_all_dist.append(corrs_hist)
    all_data.append(smpl_avg)    

#how do i know if a correlation is significant? -> shuffle all correlations over all points 
#something like that

#check if for 1 second it is the same result than for 2 seconds

plt.ion()

walls = [(-1.169,14.47),(35.33,16.47),(33.915,3.4086),(50.7458,3.1977),(49.80,16.520),(85.61,13.88),
              (85.61,-21.118),( 49.36,-22.87),(50.07,-9.905),(34.75,-9.34),(35.04,-22.576),(-0.87,-19.694)]
closed_box = walls
closed_box.append( (-1.169,14.47) ) #make sure the box is closed
closed_box = shapely.LineString( closed_box )

corrs_ampl[corrs_ampl==0] = np.nan
fig, ax = plt.subplots(figsize=(12,4))
im=ax.scatter(coords[:,0],coords[:,1],c=corrs_ampl, marker='s', s=550, cmap='binary')
plt.colorbar(im)
ax.axis('off')
#ax.plot(*closed_box.xy,color='black')
#ax.set(xlabel='Position [cm]', ylabel='Position [cm]')

corrs_angle[corrs_angle==0] = np.nan
fig, ax = plt.subplots(figsize=(12,4))
im=ax.scatter(coords[:,0],coords[:,1],c=corrs_angle, marker='s', s=550, cmap='binary')
plt.colorbar(im)
ax.axis('off')
#ax.plot(*closed_box.xy,color='black')
#ax.set(xlabel='Position [cm]', ylabel='Position [cm]')



"""
VISUALIZE SOME TRAJECTORIES
"""

file_nb1 = 131
file = open('data_model_curves/'+model+'/grid_sweep/'+data_files[file_nb1], 'rb')
data = pickle.load(file)
file.close()
samples = data['actions'] #(nb_hist,nb_samples,14,6,2)
grid_id = data_files[file_nb1].removesuffix('.pickle')
grid_id = grid_id.removeprefix('curve_')
grid_id = int(grid_id)
fig, ax = plt.subplots(nrows=1,ncols=3,figsize=(12,3))
for j in range(3):
    for i in range(samples.shape[1]-50):
        ax[j].plot(np.insert(samples[8,i,3*j,:,0],0,coords[grid_id,0]),np.insert(samples[8,i,3*j,:,1],0,coords[grid_id,1]),c='#006400')
    ax[j].plot(*closed_box.xy,color='black')
    ax[j].axis('off')
    
file_nb2 = 70
file = open('data_model_curves/'+model+'/grid_sweep/'+data_files[file_nb2], 'rb')
data = pickle.load(file)
file.close()
samples = data['actions'] #(nb_hist,nb_samples,14,6,2)
grid_id = data_files[file_nb2].removesuffix('.pickle')
grid_id = grid_id.removeprefix('curve_')
grid_id = int(grid_id)
for j in range(3):
    for i in range(samples.shape[1]-50):
        ax[j].plot(np.insert(samples[14,i,3*j,:,0],0,coords[grid_id,0]),np.insert(samples[14,i,3*j,:,1],0,coords[grid_id,1]),c='#006400')
    ax[j].plot(*closed_box.xy,color='black')
    ax[j].axis('off')
plt.savefig('figures/grid_sweep/'+model+'/'+'trajectory_TMT_'+str(file_nb1)+'_'+str(file_nb2)+'.pdf', format="pdf", bbox_inches="tight")



"""
" APPLY TSNE DIMENSIONALITY REDUCTION "
"""
#concatenate the 5 2sec average trajectories to one feature vector (60D), for each location and 10 histories (1592 vectors or so)
#then do tSNE and identify if indeed there are distinct clusters. 
#Intuitively we would expect tmt-sensitive cluster, food-sensitive, mixed,...

"""
all_data = np.vstack(all_data)
X = all_data.reshape((all_data.shape[0],60))
X_embedded = TSNE(n_components=2, learning_rate='auto',
                  init='random', perplexity=20).fit_transform(X)

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
    for j in range(5):
        c = (0,0,1,(j+1)/5)
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
    
        for j in range(5):
            c = (0,0,1,(j+1)/5)
            ax[k,i].plot(all_data[idx_,j,:,0],all_data[idx_,j,:,1],c=c)
    
plt.savefig('figures/grid_sweep/'+model+'/'+'trajectory_clusters_TMT.pdf', format="pdf", bbox_inches="tight")
plt.close()
"""