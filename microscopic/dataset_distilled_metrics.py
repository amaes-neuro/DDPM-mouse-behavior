# -*- coding: utf-8 -*-
"""
Created on Wed May  8 15:25:55 2024

Computes some distilled metrics from the dataset.
You can compare the model performance to this.

@author: ahm8208
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression


dataset_path='data/'
# read from pickle files
file = open(dataset_path+'actions.pickle', 'rb')
actions = pickle.load(file)
file.close()
file = open(dataset_path+'states.pickle', 'rb')
states = pickle.load(file)
file.close()
file = open(dataset_path+'episode_ends.pickle', 'rb')
episode_ends = pickle.load(file) # Marks one-past the last index for each episode
file.close()


#total distance covered as a function of time spent in the left side of the box
total_distance = np.zeros((len(episode_ends),2))
for i in range(len(episode_ends)):
    start_idx = 0
    if i>0:
        start_idx = episode_ends[i-1]
    total_distance[i,0] = np.sum(states[start_idx:episode_ends[i],0]<40)/(episode_ends[i]-start_idx)
    total_distance[i,1] = np.sum( np.sqrt(actions[start_idx:episode_ends[i],0]**2 + actions[start_idx:episode_ends[i],1]**2) )
    
plt.figure()
plt.scatter(total_distance[:,0],total_distance[:,1])
plt.xlabel('Fraction of time in left side of box')
plt.ylabel('Total distance covered [cm]')

#Welch's t-test on amount of distance travelled when splitting the data in two groups (no significance)
stats.ttest_ind(total_distance[np.where(total_distance[:,0]>0.5)[0],1],total_distance[np.where(total_distance[:,0]<0.5)[0],1],equal_var=False)



#violin plot of distances, one violin plot for left side and one for right side
left_side = []
right_side = []
for i in range(len(episode_ends)):
    start_idx = 0
    if i>0:
        start_idx = episode_ends[i-1]
    left = np.sum(states[start_idx:episode_ends[i],0]<40)/(episode_ends[i]-start_idx) > 0.5
    if left:
        left_side.append(np.sqrt(actions[start_idx:episode_ends[i],0]**2 + actions[start_idx:episode_ends[i],1]**2))
    else:
        right_side.append(np.sqrt(actions[start_idx:episode_ends[i],0]**2 + actions[start_idx:episode_ends[i],1]**2))

left = np.hstack(left_side)
right = np.hstack(right_side)
"""
plt.figure()
plt.violinplot([left,right],showmeans=True)
plt.xlabel('Data split according to fraction of time spent in left side of box')
plt.ylabel('Distribution of distances travelled at 3Hz [cm]')
"""

#remove zeros and plot the distribution of log-values
idx_left = np.where(left!=0)
idx_right = np.where(right!=0)
plt.figure()
plt.violinplot([np.log(left[idx_left[0]]),np.log(right[idx_right[0]])],showmeans=True)
plt.xlabel('Data split according to fraction of time spent in left side of box')
plt.ylabel('Log distribution of distances travelled at 3Hz [cm]')



#split data in two time bins and compute total distance travelled in each time bin as a function of time spent in left side
total_distance = np.zeros((len(episode_ends),4))
for i in range(len(episode_ends)):
    start_idx = 0
    if i>0:
        start_idx = episode_ends[i-1]
    food_distances = np.sqrt( states[start_idx:episode_ends[i],0]**2 + states[start_idx:episode_ends[i],1]**2 )<40
    total_distance[i,0] = np.sum(food_distances[0:int(len(food_distances)/2)])/(int(len(food_distances)/2))
    total_distance[i,1] = np.sum( np.sqrt(actions[start_idx:start_idx+int(len(food_distances)/2),0]**2 + actions[start_idx:start_idx+int(len(food_distances)/2),1]**2) )
    total_distance[i,2] = np.sum(food_distances[int(len(food_distances)/2):len(food_distances)])/(int(len(food_distances)/2))
    total_distance[i,3] = np.sum( np.sqrt(actions[start_idx+int(len(food_distances)/2):episode_ends[i],0]**2 + actions[start_idx+int(len(food_distances)/2):episode_ends[i],1]**2) )


idx1 = np.where(total_distance[:,0]>0.5)[0]
reg_1 = LinearRegression().fit(total_distance[idx1,0].reshape(-1,1), total_distance[idx1,1])
idx2 = np.where(total_distance[:,2]>0.5)[0]
reg_2 = LinearRegression().fit(total_distance[idx2,2].reshape(-1,1), total_distance[idx2,3])
idx3 = np.where(total_distance[:,0]<0.5)[0]
reg_3 = LinearRegression().fit(total_distance[idx3,0].reshape(-1,1), total_distance[idx3,1])
idx4 = np.where(total_distance[:,2]<0.5)[0]
reg_4 = LinearRegression().fit(total_distance[idx4,2].reshape(-1,1), total_distance[idx4,3])

plt.figure()
plt.scatter(total_distance[:,0],total_distance[:,1],c='blue')
plt.scatter(total_distance[:,2],total_distance[:,3],c='red')
plt.plot(np.linspace(0.6,1,20),reg_1.coef_*np.linspace(0.6,1,20)+reg_1.intercept_,c='blue')
plt.plot(np.linspace(0.6,1,20),reg_2.coef_*np.linspace(0.6,1,20)+reg_2.intercept_,c='red')
plt.plot(np.linspace(0.,0.4,20),reg_3.coef_*np.linspace(0.,0.4,20)+reg_3.intercept_,c='blue')
plt.plot(np.linspace(0.,0.4,20),reg_4.coef_*np.linspace(0.,0.4,20)+reg_4.intercept_,c='red')
plt.xlabel('Fraction of time in left side of box')
plt.ylabel('Total distance covered [cm]')



#Total distance covered as a function of the amount of crossings between left and right sides (also split in two time bins)
total_distance_crossings = np.zeros((len(episode_ends),4))
for i in range(len(episode_ends)):
    start_idx = 0
    if i>0:
        start_idx = episode_ends[i-1]
    crossings = 1.*(states[start_idx:episode_ends[i],0]<40)
    nb_crossings = abs(crossings[1:]-crossings[0:-1])
    total_distance_crossings[i,0] = np.sum(nb_crossings[0:int(len(crossings)/2)])
    total_distance_crossings[i,1] = np.sum( np.sqrt(actions[start_idx:start_idx+int(len(crossings)/2),0]**2 + actions[start_idx:start_idx+int(len(crossings)/2),1]**2) )
    total_distance_crossings[i,2] = np.sum(nb_crossings[int(len(crossings)/2):len(crossings)])
    total_distance_crossings[i,3] = np.sum( np.sqrt(actions[start_idx+int(len(crossings)/2):episode_ends[i],0]**2 + actions[start_idx+int(len(crossings)/2):episode_ends[i],1]**2) )

plt.figure()
plt.scatter(total_distance_crossings[:,0],total_distance_crossings[:,1],c='blue')
plt.scatter(total_distance_crossings[:,2],total_distance_crossings[:,3],c='red')
plt.xlabel('Number of crossings between left/right sides of cage')
plt.ylabel('Total distance covered [cm]')



#split the data in three, for each experimental phase

idx1 = np.where(total_distance[0:int(len(episode_ends)/3),0]>0.5)[0]
reg_1 = LinearRegression().fit(total_distance[idx1,0].reshape(-1,1), total_distance[idx1,1])
idx2 = np.where(total_distance[0:int(len(episode_ends)/3),2]>0.5)[0]
reg_2 = LinearRegression().fit(total_distance[idx2,2].reshape(-1,1), total_distance[idx2,3])
idx3 = np.where(total_distance[0:int(len(episode_ends)/3),0]<0.5)[0]
reg_3 = LinearRegression().fit(total_distance[idx3,0].reshape(-1,1), total_distance[idx3,1])
idx4 = np.where(total_distance[0:int(len(episode_ends)/3),2]<0.5)[0]
reg_4 = LinearRegression().fit(total_distance[idx4,2].reshape(-1,1), total_distance[idx4,3])

plt.figure()
plt.scatter(total_distance[0:int(len(episode_ends)/3),0],total_distance[0:int(len(episode_ends)/3),1],c='blue')
plt.scatter(total_distance[0:int(len(episode_ends)/3),2],total_distance[0:int(len(episode_ends)/3),3],c='red')
plt.plot(np.linspace(0.6,1,20),reg_1.coef_*np.linspace(0.6,1,20)+reg_1.intercept_,c='blue')
plt.plot(np.linspace(0.6,1,20),reg_2.coef_*np.linspace(0.6,1,20)+reg_2.intercept_,c='red')
plt.plot(np.linspace(0.,0.4,20),reg_3.coef_*np.linspace(0.,0.4,20)+reg_3.intercept_,c='blue')
plt.plot(np.linspace(0.,0.4,20),reg_4.coef_*np.linspace(0.,0.4,20)+reg_4.intercept_,c='red')
plt.xlabel('Fraction of time in left side of box')
plt.ylabel('Total distance covered [cm]')
plt.title('Phase A')

idx1 = np.where(total_distance[int(len(episode_ends)/3):int(2*len(episode_ends)/3),0]>0.5)[0]
reg_1 = LinearRegression().fit(total_distance[idx1,0].reshape(-1,1), total_distance[idx1,1])
idx2 = np.where(total_distance[int(len(episode_ends)/3):int(2*len(episode_ends)/3),2]>0.5)[0]
reg_2 = LinearRegression().fit(total_distance[idx2,2].reshape(-1,1), total_distance[idx2,3])
idx3 = np.where(total_distance[int(len(episode_ends)/3):int(2*len(episode_ends)/3),0]<0.5)[0]
reg_3 = LinearRegression().fit(total_distance[idx3,0].reshape(-1,1), total_distance[idx3,1])
idx4 = np.where(total_distance[int(len(episode_ends)/3):int(2*len(episode_ends)/3),2]<0.5)[0]
reg_4 = LinearRegression().fit(total_distance[idx4,2].reshape(-1,1), total_distance[idx4,3])

plt.figure()
plt.scatter(total_distance[int(len(episode_ends)/3):int(2*len(episode_ends)/3),0],total_distance[int(len(episode_ends)/3):int(2*len(episode_ends)/3),1],c='blue')
plt.scatter(total_distance[int(len(episode_ends)/3):int(2*len(episode_ends)/3),2],total_distance[int(len(episode_ends)/3):int(2*len(episode_ends)/3),3],c='red')
plt.plot(np.linspace(0.6,1,20),reg_1.coef_*np.linspace(0.6,1,20)+reg_1.intercept_,c='blue')
plt.plot(np.linspace(0.6,1,20),reg_2.coef_*np.linspace(0.6,1,20)+reg_2.intercept_,c='red')
plt.plot(np.linspace(0.,0.4,20),reg_3.coef_*np.linspace(0.,0.4,20)+reg_3.intercept_,c='blue')
plt.plot(np.linspace(0.,0.4,20),reg_4.coef_*np.linspace(0.,0.4,20)+reg_4.intercept_,c='red')
plt.xlabel('Fraction of time in left side of box')
plt.ylabel('Total distance covered [cm]')
plt.title('Phase B')

idx1 = np.where(total_distance[int(2*len(episode_ends)/3):,0]>0.5)[0]
reg_1 = LinearRegression().fit(total_distance[idx1,0].reshape(-1,1), total_distance[idx1,1])
idx2 = np.where(total_distance[int(2*len(episode_ends)/3):,2]>0.5)[0]
reg_2 = LinearRegression().fit(total_distance[idx2,2].reshape(-1,1), total_distance[idx2,3])
idx3 = np.where(total_distance[int(2*len(episode_ends)/3):,0]<0.5)[0]
reg_3 = LinearRegression().fit(total_distance[idx3,0].reshape(-1,1), total_distance[idx3,1])
idx4 = np.where(total_distance[int(2*len(episode_ends)/3):,2]<0.5)[0]
reg_4 = LinearRegression().fit(total_distance[idx4,2].reshape(-1,1), total_distance[idx4,3])

plt.figure()
plt.scatter(total_distance[int(2*len(episode_ends)/3):,0],total_distance[int(2*len(episode_ends)/3):,1],c='blue')
plt.scatter(total_distance[int(2*len(episode_ends)/3):,2],total_distance[int(2*len(episode_ends)/3):,3],c='red')
plt.plot(np.linspace(0.6,1,20),reg_1.coef_*np.linspace(0.6,1,20)+reg_1.intercept_,c='blue')
plt.plot(np.linspace(0.6,1,20),reg_2.coef_*np.linspace(0.6,1,20)+reg_2.intercept_,c='red')
plt.plot(np.linspace(0.,0.4,20),reg_3.coef_*np.linspace(0.,0.4,20)+reg_3.intercept_,c='blue')
plt.plot(np.linspace(0.,0.4,20),reg_4.coef_*np.linspace(0.,0.4,20)+reg_4.intercept_,c='red')
plt.xlabel('Fraction of time in left side of box')
plt.ylabel('Total distance covered [cm]')
plt.title('Phase C')
