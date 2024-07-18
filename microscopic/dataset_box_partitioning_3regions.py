# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 15:38:52 2024

Discretize the box in 3 regions, more coarse grained than the other analysis with 11 regions.
I see this as a potential way to bridge between data and model. It might show some interesting modulation of TMT or food presence.

@author: ahm8208
"""

import shapely
import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib.path import Path


def load_data(concentration, phase):
    #load data files
    path = 'C:/Users/ahm8208/OneDrive - Northwestern University/Documents/behavior_diffusion_project/data_rotated_shifted'
    file = open(path+'/points_nose_'+phase+'_'+concentration+'.pickle', 'rb')
    points = pickle.load(file)
    file.close()
    file = open(path+'/walls_'+phase+'_'+concentration+'.pickle', 'rb')
    walls = pickle.load(file)
    file.close()
    
    return points, walls


def clip_to_box(point, box):
    walls = list(map(tuple,box))
    walls.append( (box[0,0],box[0,1]) ) #make sure the box is closed
    walls = shapely.LineString( walls )
    box_path = Path( box )
    for i in range(len(point)):
        if box_path.contains_point( point[i] ) == False :
            projection = shapely.ops.nearest_points(walls,shapely.Point((point[i,0],point[i,1])))
            point[i] = np.array([projection[0].x,projection[0].y])
    return point


#I don't know if there is an elegant fast way to partition the box the way I want
#I do it manually, double check that it is correct!
def partition_box(box):
    partition = []
    
    partition.append( np.vstack( ( box[11],box[10],box[9],box[2],box[1],box[0] ) ) )
    partition.append( np.vstack( ( box[2],box[3],box[8],box[9] ) ) )
    partition.append( np.vstack( ( box[3],box[4],box[5],box[6],box[7],box[8] ) ) )
    
    return partition
    
    

#baseline transition matrix, phase A
subsample = 10
concentrations = ['1','3','10','30','90']
N = 3
transition_matrix_A = np.zeros((N,N))
for j in range(5):
    points,walls = load_data(concentrations[j],'A')
    for i in range(len(points)):
        box = walls[i]
        agent_location = points[i][0::subsample]
        agent_location = clip_to_box(agent_location, box)
        partition = partition_box(box)
        state_sequence = np.zeros((len(agent_location),))
        for h in range(N):
            box_path = Path(partition[h]).contains_points(agent_location)
            state_sequence[np.where(box_path==True)] = h
        for (k,l) in zip(state_sequence,state_sequence[1:]):
            transition_matrix_A[int(k),int(l)]+=1    

transition_matrix_A = (transition_matrix_A.T/np.sum(transition_matrix_A,axis=1)).T
plt.figure()
plt.imshow(transition_matrix_A)
plt.colorbar()
plt.xlabel('Post')
plt.ylabel('Pre')
plt.title('Phase A')


#transition matrix, phase B
subsample = 10
transition_matrix_B = np.zeros((N,N,5))
for j in range(5):
    points,walls = load_data(concentrations[j],'B')
    for i in range(len(points)):
        box = walls[i]
        agent_location = points[i][0::subsample]
        agent_location = clip_to_box(agent_location, box)
        partition = partition_box(box)
        state_sequence = np.zeros((len(agent_location),))
        for h in range(N):
            box_path = Path(partition[h]).contains_points(agent_location)
            state_sequence[np.where(box_path==True)] = h
        for (k,l) in zip(state_sequence,state_sequence[1:]):
            transition_matrix_B[int(k),int(l),j]+=1    

    transition_matrix_B[:,:,j] = (transition_matrix_B[:,:,j].T/np.sum(transition_matrix_B[:,:,j],axis=1)).T
    plt.figure()
    plt.imshow(transition_matrix_B[:,:,j]-transition_matrix_A)
    plt.colorbar()
    plt.xlabel('Post')
    plt.ylabel('Pre')
    plt.title('Phase B, TMT concentration '+concentrations[j]+'$\mu$L')



#transition matrix, phase C
subsample = 10
transition_matrix_C = np.zeros((N,N,5))
for j in range(5):
    points,walls = load_data(concentrations[j],'C')
    for i in range(len(points)):
        box = walls[i]
        agent_location = points[i][0::subsample]
        agent_location = clip_to_box(agent_location, box)
        partition = partition_box(box)
        state_sequence = np.zeros((len(agent_location),))
        for h in range(N):
            box_path = Path(partition[h]).contains_points(agent_location)
            state_sequence[np.where(box_path==True)] = h
        for (k,l) in zip(state_sequence,state_sequence[1:]):
            transition_matrix_C[int(k),int(l),j]+=1    

    transition_matrix_C[:,:,j] = (transition_matrix_C[:,:,j].T/np.sum(transition_matrix_C[:,:,j],axis=1)).T
    plt.figure()
    plt.imshow(transition_matrix_C[:,:,j]-transition_matrix_A)
    plt.colorbar()
    plt.xlabel('Post')
    plt.ylabel('Pre')
    plt.title('Phase C, TMT concentration '+concentrations[j]+'$\mu$L')



"""
Analysis of eigenvalues and corresponding decay time scale.
We compare the Markovian and non-Markovian.
"""

#phase A
subsample = np.arange(10,810,10)
transition_matrix_A_nonMarkov = np.zeros((N,N,len(subsample)))
for m in range(len(subsample)):
    for j in range(5):
        points,walls = load_data(concentrations[j],'A')
        for i in range(len(points)):
            box = walls[i]
            agent_location = points[i][0::subsample[m]]
            agent_location = clip_to_box(agent_location, box)
            partition = partition_box(box)
            state_sequence = np.zeros((len(agent_location),))
            for h in range(N):
                box_path = Path(partition[h]).contains_points(agent_location)
                state_sequence[np.where(box_path==True)] = h
            for (k,l) in zip(state_sequence,state_sequence[1:]):
                transition_matrix_A_nonMarkov[int(k),int(l),m]+=1    
        
    transition_matrix_A_nonMarkov[:,:,m] = (transition_matrix_A_nonMarkov[:,:,m].T/np.sum(transition_matrix_A_nonMarkov[:,:,m],axis=1)).T

eigs2_A_nonMarkov = np.zeros((len(subsample),))
eigs2_A = np.zeros((len(subsample),))
colors = ['rosybrown','lightcoral','indianred','brown','darkred']
for i in range(len(subsample)):
    a,b=np.linalg.eig(np.linalg.matrix_power(transition_matrix_A,i+1))
    a.sort()
    eigs2_A[i] = a[-2]
    a,b=np.linalg.eig(transition_matrix_A_nonMarkov[:,:,i])
    a.sort()
    eigs2_A_nonMarkov[i] = a[-2]

plt.figure()
plt.plot(subsample/30, eigs2_A,color=colors[4],linestyle='--', label='Markov')
plt.plot(subsample/30, eigs2_A_nonMarkov,color=colors[4], label='Non-Markov')
plt.xscale('log')
plt.ylim(0,1)
plt.xlabel('Time [s]')
plt.ylabel('$\lambda_2$')
plt.legend()
plt.title('Phase A')

#phase B
transition_matrix_B_nonMarkov = np.zeros((N,N,5,len(subsample)))
for m in range(len(subsample)):
    for j in range(5):
        points,walls = load_data(concentrations[j],'B')
        for i in range(len(points)):
            box = walls[i]
            agent_location = points[i][0::subsample[m]]
            agent_location = clip_to_box(agent_location, box)
            partition = partition_box(box)
            state_sequence = np.zeros((len(agent_location),))
            for h in range(N):
                box_path = Path(partition[h]).contains_points(agent_location)
                state_sequence[np.where(box_path==True)] = h
            for (k,l) in zip(state_sequence,state_sequence[1:]):
                transition_matrix_B_nonMarkov[int(k),int(l),j,m]+=1    
        
        transition_matrix_B_nonMarkov[:,:,j,m] = (transition_matrix_B_nonMarkov[:,:,j,m].T/np.sum(transition_matrix_B_nonMarkov[:,:,j,m],axis=1)).T

eigs2_B_nonMarkov = np.zeros((5,len(subsample)))
eigs2_B = np.zeros((5,len(subsample)))
plt.figure()
for j in range(5):
    for i in range(len(subsample)):
        a,b=np.linalg.eig(np.linalg.matrix_power(transition_matrix_B[:,:,j],i+1))
        a.sort()
        eigs2_B[j,i] = a[-2]
        a,b=np.linalg.eig(transition_matrix_B_nonMarkov[:,:,j,i])
        a.sort()
        eigs2_B_nonMarkov[j,i] = a[-2]
    
    
    plt.plot(subsample/30, eigs2_B[j,:],color=colors[j],linestyle='--')
    plt.plot(subsample/30, eigs2_B_nonMarkov[j,:],color=colors[j])
    plt.ylim(0,1)
    plt.xscale('log')
    plt.xlabel('Time [s]')
    plt.ylabel('$\lambda_2$')
    plt.title('Phase B')

#phase C
transition_matrix_C_nonMarkov = np.zeros((N,N,5,len(subsample)))
for m in range(len(subsample)):
    for j in range(5):
        points,walls = load_data(concentrations[j],'C')
        for i in range(len(points)):
            box = walls[i]
            agent_location = points[i][0::subsample[m]]
            agent_location = clip_to_box(agent_location, box)
            partition = partition_box(box)
            state_sequence = np.zeros((len(agent_location),))
            for h in range(N):
                box_path = Path(partition[h]).contains_points(agent_location)
                state_sequence[np.where(box_path==True)] = h
            for (k,l) in zip(state_sequence,state_sequence[1:]):
                transition_matrix_C_nonMarkov[int(k),int(l),j,m]+=1    
        
        transition_matrix_C_nonMarkov[:,:,j,m] = (transition_matrix_C_nonMarkov[:,:,j,m].T/np.sum(transition_matrix_C_nonMarkov[:,:,j,m],axis=1)).T

eigs2_C_nonMarkov = np.zeros((5,len(subsample)))
eigs2_C = np.zeros((5,len(subsample)))
plt.figure()
for j in range(5):
    for i in range(len(subsample)):
        a,b=np.linalg.eig(np.linalg.matrix_power(transition_matrix_C[:,:,j],i+1))
        a.sort()
        eigs2_C[j,i] = a[-2]
        a,b=np.linalg.eig(transition_matrix_C_nonMarkov[:,:,j,i])
        a.sort()
        eigs2_C_nonMarkov[j,i] = a[-2]
    
    
    plt.plot(subsample/30, eigs2_C[j,:],color=colors[j],linestyle='--')
    plt.plot(subsample/30, eigs2_C_nonMarkov[j,:],color=colors[j])
    plt.ylim(0,1)
    plt.xscale('log')
    plt.xlabel('Time [s]')
    plt.ylabel('$\lambda_2$')
    plt.title('Phase C')


#time scales
plt.figure()
plt.hlines(-1/np.log(eigs2_A[0])/3,0,4,linestyles='--',label='Phase A')
plt.plot(-1/np.log(eigs2_B[:,0])/3,label='Phase B')
plt.plot(-1/np.log(eigs2_C[:,0])/3,label='Phase C')
plt.xticks(np.arange(5),['1','3','10','30','90'])
plt.xlabel('TMT concentration [$\mu$L]')
plt.ylabel('Time constant decay [s]')
plt.legend()
