# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 10:27:38 2024

Visualizing the condition distributions

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
#import cv2
import matplotlib.pyplot as plt
import os
import shapely


walls = [(-0.90,13.0),(35.53,14.88),(34.11,1.88),(51.00,1.68),(50.03,14.94),(85.83,12.31),
              (85.83,-22.69),(49.54,-24.40),(50.23,-11.48),(35.01,-10.94),(35.23,-24.07),(-0.66,-21.24)] #this is static, the box does not change over time
closed_box = walls
closed_box.append( (-0.90,13.0) ) #make sure the box is closed
closed_box = shapely.LineString( closed_box )


def compute_sensory_field(location):
    nb_angles = 18
    sensory_field = np.zeros((nb_angles,))
    for j in range(nb_angles):
        line = shapely.LineString([ (location[0],location[1]),
                                   (location[0]+1e3*np.cos(j*2*np.pi/nb_angles),location[1]+1e3*np.sin(j*2*np.pi/nb_angles)) ])
        inters = shapely.intersection(line, closed_box)
        if inters.geom_type == 'MultiPoint': #multiple intersections
            temps = np.zeros((len(inters.geoms),))        
            for i in range(len(inters.geoms)):
                temps[i] = np.sqrt( (inters.geoms[i].x-location[0])**2 + (inters.geoms[i].y-location[1])**2 )
            sensory_field[j] = np.min(temps) 
        elif inters.geom_type == 'LineString': #this happens when the location is outside the walls
            sensory_field[j] = 0
        else: #a single intersection
            sensory_field[j] = np.sqrt( (inters.x-location[0])**2 + (inters.y-location[1])**2 )
    return sensory_field


#load trained model
path = 'checkpoints/t3.pt'
if torch.cuda.is_available():
    state_dict = torch.load(path, map_location='cuda')
else:
    state_dict = torch.load(path, map_location='cpu')

#parameters
pred_horizon = 4
obs_horizon = 1
action_horizon = 1
obs_dim = 27
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
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
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
max_steps = 27000/subsample #this is for fifteen minutes of simulates as the videos were recorded at 30Hz
env = Mouse2DEnv(render_mode='rgb_array')
# use a seed >200 to avoid initial states seen in the training dataset
env.seed(2000)


#choose obs
location = np.array([60,-5])
food = 0
threat = 0
sensory = compute_sensory_field(location)
#obs = np.hstack((location,hunger,threat,sensory))
# get first observation, conditioned on threat level (TMT concentration)
obs, info = env.reset(location=location,direction=np.array([0,0]),side=np.array([0,0]),
                      time=0,food=food,threat=threat)


# keep a queue of last steps of observations
obs_deque = collections.deque(
    [obs] * obs_horizon, maxlen=obs_horizon)
        

samples = 100
trajectory = np.zeros((samples,action_horizon+1,2))
for j in tqdm(range(samples)):
    trajectory[j,0,0] = obs[0]
    trajectory[j,0,1] = obs[1]
    B = 1
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
    
    # execute action_horizon number of steps
    # without replanning
    for i in range(len(action)):
        # stepping env
        obs, reward, done, _, info = env.step(action[i])
        if j==0:
            obs_deque.append(obs)
        trajectory[j,i+1,0] = obs[0]
        trajectory[j,i+1,1] = obs[1]
    obs, info = env.reset(location=np.array([trajectory[0,len(action),0],trajectory[0,len(action),1]]),food=food,threat=threat)#0.4*j/samples)


#visualize trajectories
from matplotlib import cm
plt.figure()
for j in range(samples):
    plt.plot(trajectory[j,:,0],trajectory[j,:,1],c=cm.hot(j/samples))
plt.plot(*closed_box.xy)    




