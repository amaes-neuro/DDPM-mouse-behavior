# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 15:39:26 2024

Choose a coordinate and run a time sweep.

@author: amaes
"""


import pickle
import torch
from ConditionalDiffuser import ConditionalUnet1D
from mouse_2D import Mouse2DEnv
import collections
from tqdm.auto import tqdm
import numpy as np
from MouseTrajectoryDataset import normalize_data, unnormalize_data, MouseTrajectoryDataset
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import os
import sys

#load trained model
model = sys.argv[1]
dataset_name = sys.argv[2]
idx = sys.argv[3]
sample = sys.argv[4]

path = 'checkpoints/'+model+'.pt'
if torch.cuda.is_available():
    state_dict = torch.load(path, map_location='cuda')
else:
    state_dict = torch.load(path, map_location='cpu')
if not os.path.exists('data_model_curves/'+model+'/samples_timesweep_'+idx):
    os.makedirs('data_model_curves/'+model+'/samples_timesweep_'+idx)

#parameters
pred_horizon = 4
obs_horizon = 1
action_horizon = 1
obs_dim = 8
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
    action_horizon=action_horizon,
    name=dataset_name
)
# save training data statistics (min, max) for each dim
stats = dataset.stats

#noise scheduler
num_diffusion_iters = 50
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

#set up loop
threat_values = [-0.2,-0.1,0,0.1,0.2,0.3,0.4]
time_steps = 6
env = Mouse2DEnv(render_mode='rgb_array')

file = open('data/'+dataset_name+'_coordinates.pickle', 'rb')
coords = pickle.load(file)
file.close()

nb_timeshifts = 45
curves = np.zeros((nb_timeshifts,2*len(threat_values),time_steps,2))

for h in range(nb_timeshifts):       
    
    location = coords[int(idx),0:2]
    loc_avg = coords[int(idx),2:4]

    for i in range(2*len(threat_values)):    
        
        threat = threat_values[i%len(threat_values)]
        if i>=len(threat_values):
            food = 1
        else:
            food = 0    
        
        #set up state
        obs, info = env.reset(location = location, loc_avg=loc_avg,  time = h*2700/nb_timeshifts, food = food, threat = threat)
        done = False
        step_idx = 0
        # keep a queue of last steps of observations
        obs_deque = collections.deque(
            [obs] * obs_horizon, maxlen=obs_horizon)
        trajectory=[]
        with tqdm(total=time_steps, desc="Eval Mouse2DEnv") as pbar:
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
        
                # stepping env
                obs, reward, done, _, info = env.step(action[0])
                # save observations
                obs_deque.append(obs)
                trajectory.append(obs)
        
                # update progress bar
                step_idx += 1
                pbar.update(1)
                pbar.set_postfix(reward=reward)
                if step_idx >= time_steps:
                    done = True
                    env.close()
                if done:
                    break
        
        #save trajectory
        curves[h,i,:,:] = np.vstack(trajectory)[:,0:2]
        
#save curve    
data_dict = {'state': coords[int(idx),:], 'actions': curves}    
print('Save sampled actions in state:',data_dict['state'])
with open('data_model_curves/'+model+'/samples_timesweep_'+idx+'/curve_'+sample+'.pickle', 'wb') as file:
    pickle.dump(data_dict, file)



