# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:05:42 2024

Run a trained model many times to extract statistics.

@author: ahm8208
"""

import torch
from ConditionalDiffuser import ConditionalUnet1D
from mouse_2D import Mouse2DEnv
import collections
from tqdm.auto import tqdm
import numpy as np
from MouseTrajectoryDataset import normalize_data, unnormalize_data, MouseTrajectoryDataset
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pickle

#load trained model
path = 'checkpoints/test31.pt'
state_dict = torch.load(path, map_location='cuda')

#parameters
pred_horizon = 16
obs_horizon = 12
action_horizon = 4
obs_dim = 23
action_dim = 2

# create network object
noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*obs_horizon
)

ema_noise_pred_net = noise_pred_net
ema_noise_pred_net.load_state_dict(state_dict)
print('Weights loaded')

# device transfer
device = torch.device('cuda')
_ = noise_pred_net.to(device)

#get stats
# create dataset from file
dataset = MouseTrajectoryDataset(
    dataset_path='data/',
    pred_horizon=pred_horizon,
    obs_horizon=obs_horizon,
    action_horizon=action_horizon
)
# save training data statistics (min, max) for each dim
stats = dataset.stats

#noise scheduler
num_diffusion_iters = 100
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    # the choise of beta schedule has big impact on performance
    # we found squared cosine works the best
    beta_schedule='squaredcos_cap_v2',
    # clip output to [-1,1] to improve stability
    clip_sample=True,
    # our network predicts noise (instead of denoised action)
    prediction_type='epsilon'
)

# limit enviornment interaction before termination
subsample = 10
max_steps = int(27000/subsample) #this is for fifteen minutes of simulates as the videos were recorded at 30Hz
env = Mouse2DEnv(render_mode='rgb_array')
# use a seed >200 to avoid initial states seen in the training dataset
env.seed(10000)


nb_its = 30
trajectories = np.zeros((nb_its*max_steps,2))
for p in range(nb_its):

    # get first observation, conditioned on threat level (TMT concentration)
    threat = 0.05+0.9*p/nb_its
    obs, info = env.reset(threat=threat)
    
    # keep a queue of last steps of observations
    obs_deque = collections.deque(
        [obs] * obs_horizon, maxlen=obs_horizon)
    # save visualization and rewards
    imgs = [env.render()]
    trajectory = [obs]
    rewards = list()
    done = False
    step_idx = 0
    
    with tqdm(total=max_steps, desc="Eval Mouse2DEnv") as pbar:
        while not done:
            B = 1
            # stack the last obs_horizon number of observations
            obs_seq = np.stack(obs_deque)
            # normalize observation
            nobs = normalize_data(obs_seq, stats=stats['obs'])
            # device transfer
            nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)
    
            # infer action
            with torch.no_grad():
                # reshape observation to (B,obs_horizon*obs_dim)
                obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)
    
                # initialize action from Guassian noise
                noisy_action = torch.randn(
                    (B, pred_horizon, action_dim), device=device)
                naction = noisy_action
    
                # init scheduler
                noise_scheduler.set_timesteps(num_diffusion_iters)
    
                for k in noise_scheduler.timesteps:
                    # predict noise
                    noise_pred = ema_noise_pred_net(
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond
                    )
    
                    # inverse diffusion step (remove noise)
                    naction = noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=naction
                    ).prev_sample
    
            # unnormalize action
            naction = naction.detach().to('cpu').numpy()
            # (B, pred_horizon, action_dim)
            naction = naction[0]
            action_pred = unnormalize_data(naction, stats=stats['action'])
    
            # only take action_horizon number of actions
            start = obs_horizon - 1
            end = start + action_horizon
            action = action_pred[start:end,:]
            # (action_horizon, action_dim)
    
            # execute action_horizon number of steps
            # without replanning
            for i in range(len(action)):
                # stepping env
                obs, reward, done, _, info = env.step(action[i])
                # save observations
                obs_deque.append(obs)
                # and reward/vis
                trajectory.append(obs)
                rewards.append(reward)
                imgs.append(env.render())
    
                # update progress bar
                step_idx += 1
                pbar.update(1)
                pbar.set_postfix(reward=reward)
                if step_idx > max_steps:
                    done = True
                    env.close()
                if done:
                    break
    
    temp = np.vstack(trajectory)
    trajectories[p*max_steps:(p+1)*max_steps] = temp[0:max_steps,0:2]

#save the trajectories
with open('data/trajectories_test31.pickle', 'wb') as file:
    pickle.dump(np.float32(trajectories), file)

#visualize
#split data in two time bins and compute total distance travelled in each time bin as a function of time spent in left side
total_distance = np.zeros((nb_its,4))

for i in range(nb_its):
    actions = trajectories[i*max_steps+1:(i+1)*max_steps] - trajectories[i*max_steps:(i+1)*max_steps-1]
    food_distances = np.sqrt( trajectories[i*max_steps:(i+1)*max_steps,0]**2 + trajectories[i*max_steps:(i+1)*max_steps,1]**2 )<40
    total_distance[i,0] = np.sum(food_distances[0:int(len(food_distances)/2)])/(int(len(food_distances)/2))
    total_distance[i,1] = np.sum( np.sqrt(actions[:,0]**2 + actions[:,1]**2) )
    total_distance[i,2] = np.sum(food_distances[int(len(food_distances)/2):len(food_distances)])/(int(len(food_distances)/2))
    total_distance[i,3] = np.sum( np.sqrt(actions[:,0]**2 + actions[:,1]**2) )


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



#violin plot of distances, one violin plot for left side and one for right side
left_side = []
right_side = []
for i in range(nb_its):
    actions = trajectories[i*max_steps+1:(i+1)*max_steps] - trajectories[i*max_steps:(i+1)*max_steps-1]
    food_distances = np.sqrt( trajectories[i*max_steps:(i+1)*max_steps,0]**2 + trajectories[i*max_steps:(i+1)*max_steps,1]**2 )<40
    left = np.sum(food_distances)>0.5
    if left:
        left_side.append(np.sqrt(actions[:,0]**2 + actions[:,1]**2))
    else:
        right_side.append(np.sqrt(actions[:,0]**2 + actions[:,1]**2))

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

