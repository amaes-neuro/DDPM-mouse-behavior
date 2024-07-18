# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:29:35 2024

Discretize the box in 11 regions, this gives a more fine grained idea of how the mouse is traversing this space.
I see this as a potential way to bridge between data and model. It might show some interesting modulation of TMT or food presence.

@author: ahm8208
"""

import shapely
import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib.path import Path
from scipy import stats

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
    left_square = np.vstack( ( np.array([box[1][0]/3+2*box[0][0]/3,box[2][1]]),
                              np.array([2*box[1][0]/3+box[0][0]/3,box[2][1]]),
                              np.array([2*box[10][0]/3+box[11][0]/3,box[9][1]]),
                              np.array([box[10][0]/3+2*box[11][0]/3,box[9][1]])
                              ) )
    right_square = np.vstack( ( np.array([box[5][0]/3+2*box[4][0]/3,box[3][1]]),
                              np.array([2*box[5][0]/3+box[4][0]/3,box[3][1]]),
                              np.array([2*box[6][0]/3+box[7][0]/3,box[8][1]]),
                              np.array([box[6][0]/3+2*box[7][0]/3,box[8][1]])
                              ) )
    partition = []
    
    partition.append( np.vstack( ( box[11],box[10],left_square[2],left_square[3] ) ) )
    partition.append( np.vstack( ( box[11],box[0],left_square[0],left_square[3] ) ) )
    partition.append( np.vstack( ( left_square[0],left_square[1],left_square[2],left_square[3] ) ) )
    partition.append( np.vstack( ( box[0],box[1],left_square[1],left_square[0] ) ) )
    partition.append( np.vstack( ( box[1],box[2],box[9],box[10],left_square[2],left_square[1] ) ) )
    
    partition.append( np.vstack( ( box[2],box[3],box[8],box[9] ) ) )

    partition.append( np.vstack( ( box[3],box[4],right_square[0],right_square[3],box[7],box[8] ) ) )
    partition.append( np.vstack( ( box[4],box[5],right_square[1],right_square[0] ) ) )
    partition.append( np.vstack( ( right_square[0],right_square[1],right_square[2],right_square[3] ) ) )
    partition.append( np.vstack( ( box[5],box[6],right_square[2],right_square[1] ) ) )
    partition.append( np.vstack( ( box[6],box[7],right_square[3],right_square[2] ) ) )
    
    return partition
    
    

#baseline transition matrix, phase A
subsample = 10
concentrations = ['1','3','10','30','90']
transition_matrix_A = np.zeros((11,11))
for j in range(5):
    points,walls = load_data(concentrations[j],'A')
    for i in range(len(points)):
        box = walls[i]
        agent_location = points[i][0::subsample]
        agent_location = clip_to_box(agent_location, box)
        partition = partition_box(box)
        state_sequence = np.zeros((len(agent_location),))
        for h in range(11):
            box_path = Path(partition[h]).contains_points(agent_location)
            state_sequence[np.where(box_path==True)] = h
        for (k,l) in zip(state_sequence,state_sequence[1:]):
            transition_matrix_A[int(k),int(l)]+=1    

count_A = np.sum(transition_matrix_A,axis=1)
transition_matrix_A = (transition_matrix_A.T/np.sum(transition_matrix_A,axis=1)).T
plt.figure()
plt.imshow(transition_matrix_A)
plt.colorbar()
plt.xlabel('Post')
plt.ylabel('Pre')
plt.title('Phase A')


#transition matrix, phase B
subsample = 10
transition_matrix_B = np.zeros((11,11,5))
count_B = np.zeros((11,5))
for j in range(5):
    points,walls = load_data(concentrations[j],'B')
    for i in range(len(points)):
        box = walls[i]
        agent_location = points[i][0::subsample]
        agent_location = clip_to_box(agent_location, box)
        partition = partition_box(box)
        state_sequence = np.zeros((len(agent_location),))
        for h in range(11):
            box_path = Path(partition[h]).contains_points(agent_location)
            state_sequence[np.where(box_path==True)] = h
        for (k,l) in zip(state_sequence,state_sequence[1:]):
            transition_matrix_B[int(k),int(l),j]+=1    

    count_B[:,j] = np.sum(transition_matrix_B[:,:,j],axis=1)
    transition_matrix_B[:,:,j] = (transition_matrix_B[:,:,j].T/np.sum(transition_matrix_B[:,:,j],axis=1)).T
    plt.figure()
    plt.imshow(transition_matrix_B[:,:,j]-transition_matrix_A)
    plt.colorbar()
    plt.xlabel('Post')
    plt.ylabel('Pre')
    plt.title('Phase B, TMT concentration '+concentrations[j]+'$\mu$L')



#transition matrix, phase C
subsample = 10
transition_matrix_C = np.zeros((11,11,5))
count_C = np.zeros((11,5))
for j in range(5):
    points,walls = load_data(concentrations[j],'C')
    for i in range(len(points)):
        box = walls[i]
        agent_location = points[i][0::subsample]
        agent_location = clip_to_box(agent_location, box)
        partition = partition_box(box)
        state_sequence = np.zeros((len(agent_location),))
        for h in range(11):
            box_path = Path(partition[h]).contains_points(agent_location)
            state_sequence[np.where(box_path==True)] = h
        for (k,l) in zip(state_sequence,state_sequence[1:]):
            transition_matrix_C[int(k),int(l),j]+=1    

    count_C[:,j] = np.sum(transition_matrix_C[:,:,j],axis=1)
    transition_matrix_C[:,:,j] = (transition_matrix_C[:,:,j].T/np.sum(transition_matrix_C[:,:,j],axis=1)).T
    plt.figure()
    plt.imshow(transition_matrix_C[:,:,j]-transition_matrix_A)
    plt.colorbar()
    plt.xlabel('Post')
    plt.ylabel('Pre')
    plt.title('Phase C, TMT concentration '+concentrations[j]+'$\mu$L')



"""
Analysis of eigenvalues and corresponding decay time scale.
We compare the Markovian assumption with the data.
"""

#phase A
subsample = np.arange(10,810,10)
transition_matrix_A_nonMarkov = np.zeros((11,11,len(subsample)))
for m in range(len(subsample)):
    for j in range(5):
        points,walls = load_data(concentrations[j],'A')
        for i in range(len(points)):
            box = walls[i]
            agent_location = points[i][0::subsample[m]]
            agent_location = clip_to_box(agent_location, box)
            partition = partition_box(box)
            state_sequence = np.zeros((len(agent_location),))
            for h in range(11):
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
transition_matrix_B_nonMarkov = np.zeros((11,11,5,len(subsample)))
for m in range(len(subsample)):
    for j in range(5):
        points,walls = load_data(concentrations[j],'B')
        for i in range(len(points)):
            box = walls[i]
            agent_location = points[i][0::subsample[m]]
            agent_location = clip_to_box(agent_location, box)
            partition = partition_box(box)
            state_sequence = np.zeros((len(agent_location),))
            for h in range(11):
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
transition_matrix_C_nonMarkov = np.zeros((11,11,5,len(subsample)))
for m in range(len(subsample)):
    for j in range(5):
        points,walls = load_data(concentrations[j],'C')
        for i in range(len(points)):
            box = walls[i]
            agent_location = points[i][0::subsample[m]]
            agent_location = clip_to_box(agent_location, box)
            partition = partition_box(box)
            state_sequence = np.zeros((len(agent_location),))
            for h in range(11):
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



"""
In order to test non-stationarity
Approach: split trials in two time bins. Compute transition matrix for each of them.
"""

#phase A
subsample = np.arange(10,810,10)
transition_matrix_A_split = np.zeros((11,11,2,len(subsample)))
counts_A_split = np.zeros((11,2))
dwells_A_split = np.zeros((11,2))
for m in range(len(subsample)):
    for j in range(5):
        points,walls = load_data(concentrations[j],'A')
        for i in range(len(points)):
            box = walls[i]
            agent_location = points[i][0::subsample[m]]
            agent_location = clip_to_box(agent_location, box)
            partition = partition_box(box)
            state_sequence = np.zeros((len(agent_location),))
            for h in range(11):
                box_path = Path(partition[h]).contains_points(agent_location)
                state_sequence[np.where(box_path==True)] = h
            for (k,l) in zip(state_sequence[0:int(len(agent_location)/2)],state_sequence[1:int(len(agent_location)/2+1)]):
                transition_matrix_A_split[int(k),int(l),0,m]+=1    
            for (k,l) in zip(state_sequence[int(len(agent_location)/2):],state_sequence[int(len(agent_location)/2+1):]):
                transition_matrix_A_split[int(k),int(l),1,m]+=1    
    if m==0:
        counts_A_split[:,0] = np.sum(transition_matrix_A_split[:,:,0,m],axis=1)
        counts_A_split[:,1] = np.sum(transition_matrix_A_split[:,:,1,m],axis=1)
        dwells_A_split[:,0] = np.sum(transition_matrix_A_split[:,:,0,m],axis=0)
        dwells_A_split[:,1] = np.sum(transition_matrix_A_split[:,:,1,m],axis=0)
    transition_matrix_A_split[:,:,0,m] = (transition_matrix_A_split[:,:,0,m].T/np.sum(transition_matrix_A_split[:,:,0,m],axis=1)).T
    transition_matrix_A_split[:,:,1,m] = (transition_matrix_A_split[:,:,1,m].T/np.sum(transition_matrix_A_split[:,:,1,m],axis=1)).T

eigs2_A_split0 = np.zeros((len(subsample),))
eigs2_A_split1 = np.zeros((len(subsample),))
colors = ['rosybrown','lightcoral','indianred','brown','darkred']
for i in range(len(subsample)):
    a,b=np.linalg.eig(transition_matrix_A_split[:,:,0,i])
    a.sort()
    eigs2_A_split0[i] = a[-2]
    a,b=np.linalg.eig(transition_matrix_A_split[:,:,1,i])
    a.sort()
    eigs2_A_split1[i] = a[-2]

plt.figure()
plt.plot(subsample/30, eigs2_A_split0,color=colors[4],linestyle='--', label='First half')
plt.plot(subsample/30, eigs2_A_split1,color=colors[4], label='Second half')
plt.xscale('log')
plt.ylim(0,1)
plt.xlabel('Time [s]')
plt.ylabel('$\lambda_2$')
plt.legend()
plt.title('Phase A')



#phase B
subsample = np.arange(10,510,10)
transition_matrix_B_split = np.zeros((11,11,2,5,len(subsample)))
counts_B_split = np.zeros((11,2,5))
dwells_B_split = np.zeros((11,2,5))
for m in range(len(subsample)):
    for j in range(5):
        points,walls = load_data(concentrations[j],'B')
        for i in range(len(points)):
            box = walls[i]
            agent_location = points[i][0::subsample[m]]
            agent_location = clip_to_box(agent_location, box)
            partition = partition_box(box)
            state_sequence = np.zeros((len(agent_location),))
            for h in range(11):
                box_path = Path(partition[h]).contains_points(agent_location)
                state_sequence[np.where(box_path==True)] = h
            for (k,l) in zip(state_sequence[0:int(len(agent_location)/2)],state_sequence[1:int(len(agent_location)/2+1)]):
                transition_matrix_B_split[int(k),int(l),0,j,m]+=1    
            for (k,l) in zip(state_sequence[int(len(agent_location)/2):],state_sequence[int(len(agent_location)/2+1):]):
                transition_matrix_B_split[int(k),int(l),1,j,m]+=1    
        if m==0:
            counts_B_split[:,0,j] = np.sum(transition_matrix_B_split[:,:,0,j,m],axis=1)
            counts_B_split[:,1,j] = np.sum(transition_matrix_B_split[:,:,1,j,m],axis=1)
            dwells_B_split[:,0,j] = np.sum(transition_matrix_B_split[:,:,0,j,m],axis=0)
            dwells_B_split[:,1,j] = np.sum(transition_matrix_B_split[:,:,1,j,m],axis=0)
        transition_matrix_B_split[:,:,0,j,m] = (transition_matrix_B_split[:,:,0,j,m].T/np.sum(transition_matrix_B_split[:,:,0,j,m],axis=1)).T
        transition_matrix_B_split[:,:,1,j,m] = (transition_matrix_B_split[:,:,1,j,m].T/np.sum(transition_matrix_B_split[:,:,1,j,m],axis=1)).T

eigs2_B_split0 = np.zeros((5,len(subsample)))
eigs2_B_split1 = np.zeros((5,len(subsample)))
plt.figure()
for j in range(5):
    for i in range(len(subsample)):
        a,b=np.linalg.eig(transition_matrix_B_split[:,:,0,j,i])
        a.sort()
        eigs2_B_split0[j,i] = a[-2]
        a,b=np.linalg.eig(transition_matrix_B_split[:,:,1,j,i])
        a.sort()
        eigs2_B_split1[j,i] = a[-2]
    
    
    plt.plot(subsample/30, eigs2_B_split0[j,:],color=colors[j],linestyle='--')
    plt.plot(subsample/30, eigs2_B_split1[j,:],color=colors[j])
    plt.ylim(0,1)
    plt.xscale('log')
    plt.xlabel('Time [s]')
    plt.ylabel('$\lambda_2$')
    plt.title('Phase B')



#phase C
subsample = np.arange(10,210,10)
transition_matrix_C_split = np.zeros((11,11,2,5,len(subsample)))
counts_C_split = np.zeros((11,2,5))
dwells_C_split = np.zeros((11,2,5))
for m in range(len(subsample)):
    for j in range(5):
        points,walls = load_data(concentrations[j],'C')
        for i in range(len(points)):
            box = walls[i]
            agent_location = points[i][0::subsample[m]]
            agent_location = clip_to_box(agent_location, box)
            partition = partition_box(box)
            state_sequence = np.zeros((len(agent_location),))
            for h in range(11):
                box_path = Path(partition[h]).contains_points(agent_location)
                state_sequence[np.where(box_path==True)] = h
            for (k,l) in zip(state_sequence[0:int(len(agent_location)/2)],state_sequence[1:int(len(agent_location)/2+1)]):
                transition_matrix_C_split[int(k),int(l),0,j,m]+=1    
            for (k,l) in zip(state_sequence[int(len(agent_location)/2):],state_sequence[int(len(agent_location)/2+1):]):
                transition_matrix_C_split[int(k),int(l),1,j,m]+=1    
        if m==0:
            counts_C_split[:,0,j] = np.sum(transition_matrix_C_split[:,:,0,j,m],axis=1)
            counts_C_split[:,1,j] = np.sum(transition_matrix_C_split[:,:,1,j,m],axis=1)
            dwells_C_split[:,0,j] = np.sum(transition_matrix_C_split[:,:,0,j,m],axis=0)
            dwells_C_split[:,1,j] = np.sum(transition_matrix_C_split[:,:,1,j,m],axis=0)
        transition_matrix_C_split[:,:,0,j,m] = (transition_matrix_C_split[:,:,0,j,m].T/np.sum(transition_matrix_C_split[:,:,0,j,m],axis=1)).T
        transition_matrix_C_split[:,:,1,j,m] = (transition_matrix_C_split[:,:,1,j,m].T/np.sum(transition_matrix_C_split[:,:,1,j,m],axis=1)).T

eigs2_C_split0 = np.zeros((5,len(subsample)))
eigs2_C_split1 = np.zeros((5,len(subsample)))
plt.figure()
for j in range(5):
    for i in range(len(subsample)):
        a,b=np.linalg.eig(transition_matrix_C_split[:,:,0,j,i])
        a.sort()
        eigs2_C_split0[j,i] = a[-2]
        a,b=np.linalg.eig(transition_matrix_C_split[:,:,1,j,i])
        a.sort()
        eigs2_C_split1[j,i] = a[-2]
    
    
    plt.plot(subsample/30, eigs2_C_split0[j,:],color=colors[j],linestyle='--')
    plt.plot(subsample/30, eigs2_C_split1[j,:],color=colors[j])
    plt.ylim(0,1)
    plt.xscale('log')
    plt.xlabel('Time [s]')
    plt.ylabel('$\lambda_2$')
    plt.title('Phase C')
    
    
"""
Non-stationarity continued: chi-squared tests of stationarity
"""

#phase A
chi_sqs_A = np.zeros((11,2)) #one row for each state and one column for each time split
for i in range(11):
    temp0 = 0
    temp1 = 0
    for j in range(11):
        if transition_matrix_A[i,j] != 0:
            temp0 += (transition_matrix_A_split[i,j,0,0]-transition_matrix_A[i,j])**2/transition_matrix_A[i,j]
            temp1 += (transition_matrix_A_split[i,j,1,0]-transition_matrix_A[i,j])**2/transition_matrix_A[i,j]
    chi_sqs_A[i,0] = counts_A_split[i,0]*temp0
    chi_sqs_A[i,1] = counts_A_split[i,1]*temp1

print(stats.chi2.sf(chi_sqs_A,10))

#phase B
chi_sqs_B = np.zeros((11,2,5))
for k in range(5):
    for i in range(11):
        temp0 = 0
        temp1 = 0
        for j in range(11):
            if transition_matrix_B[i,j,k] != 0:
                temp0 += (transition_matrix_B_split[i,j,0,k,0]-transition_matrix_B[i,j,k])**2/transition_matrix_B[i,j,k]
                temp1 += (transition_matrix_B_split[i,j,1,k,0]-transition_matrix_B[i,j,k])**2/transition_matrix_B[i,j,k]
        chi_sqs_B[i,0,k] = counts_B_split[i,0,k]*temp0
        chi_sqs_B[i,1,k] = counts_B_split[i,1,k]*temp1

#phase C
chi_sqs_C = np.zeros((11,2,5))
for k in range(5):
    for i in range(11):
        temp0 = 0
        temp1 = 0
        for j in range(11):
            if transition_matrix_C[i,j,k] != 0:
                temp0 += (transition_matrix_C_split[i,j,0,k,0]-transition_matrix_C[i,j,k])**2/transition_matrix_C[i,j,k]
                temp1 += (transition_matrix_C_split[i,j,1,k,0]-transition_matrix_C[i,j,k])**2/transition_matrix_C[i,j,k]
        chi_sqs_C[i,0,k] = counts_C_split[i,0,k]*temp0
        chi_sqs_C[i,1,k] = counts_C_split[i,1,k]*temp1

#I do not see any evidence of non-stationarity in the transition probabilities.
#Is there non-stationarity in the dwell probability?
print(stats.chisquare(dwells_A_split[:,0],dwells_A_split[:,1]*(np.sum(dwells_A_split[:,0])/np.sum(dwells_A_split[:,1]))))


for i in range(5):
    print(stats.chisquare(dwells_B_split[:,0,i],dwells_B_split[:,1,i]*(np.sum(dwells_B_split[:,0,i])/np.sum(dwells_B_split[:,1,i]))))
    print(stats.chisquare(dwells_C_split[:,0,i],dwells_C_split[:,1,i]*(np.sum(dwells_C_split[:,0,i])/np.sum(dwells_C_split[:,1,i]))))

#If this is correct, there seems to be non-stationarity in the state distribution for all phases
#Is this also the case for more coarse-graining? 
dwells_A_split_coarse = np.zeros((3,2))
for i in range(2):
    dwells_A_split_coarse[0,i] = np.sum(dwells_A_split[0:5,i])
    dwells_A_split_coarse[1,i] = np.sum(dwells_A_split[5,i])
    dwells_A_split_coarse[2,i] = np.sum(dwells_A_split[6:,i])

print(stats.chisquare(dwells_A_split_coarse[:,0],dwells_A_split_coarse[:,1]*(np.sum(dwells_A_split_coarse[:,0])/np.sum(dwells_A_split_coarse[:,1]))))

dwells_B_split_coarse = np.zeros((3,2,5))
for j in range(5):
    for i in range(2):
        dwells_B_split_coarse[0,i,j] = np.sum(dwells_B_split[0:5,i,j])
        dwells_B_split_coarse[1,i,j] = np.sum(dwells_B_split[5,i,j])
        dwells_B_split_coarse[2,i,j] = np.sum(dwells_B_split[6:,i,j])
        
dwells_C_split_coarse = np.zeros((3,2,5))
for j in range(5):
    for i in range(2):
        dwells_C_split_coarse[0,i,j] = np.sum(dwells_C_split[0:5,i,j])
        dwells_C_split_coarse[1,i,j] = np.sum(dwells_C_split[5,i,j])
        dwells_C_split_coarse[2,i,j] = np.sum(dwells_C_split[6:,i,j])

#conclusion: phase A coarse grained is not-significantly different in the two time bins
#from phase B onwards there are significant differences between the two time bins, this is also true if you combine all concentration data (as below)
dwells_B = np.zeros((3,2))
for i in range(2):
    dwells_B[0,i] = np.sum(dwells_B_split[0:5,i,:])
    dwells_B[1,i] = np.sum(dwells_B_split[5,i,:])
    dwells_B[2,i] = np.sum(dwells_B_split[6:,i,:])

print(stats.chisquare(dwells_B[:,0],dwells_B[:,1]*(np.sum(dwells_B[:,0])/np.sum(dwells_B[:,1]))))



