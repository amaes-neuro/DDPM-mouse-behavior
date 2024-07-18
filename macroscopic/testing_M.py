# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 17:01:03 2024

@author: ahm8208
"""

import torch
from ConditionalDiffuser import ConditionalUnet1D
from mouse_M import MouseMEnv
import collections
from tqdm.auto import tqdm
import numpy as np
from MouseTrajectoryDataset_M import normalize_data, unnormalize_data, MouseTrajectoryDataset
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import matplotlib.pyplot as plt

#load trained model
path = 'checkpoints/t_M_3.pt'
if torch.cuda.is_available():
    state_dict = torch.load(path, map_location='cuda')
else:
    state_dict = torch.load(path, map_location='cpu')

#parameters
pred_horizon = 4
obs_horizon = 1
action_horizon = 1
obs_dim = 5
action_dim = 1

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
subsample = 30
max_steps = 27000/subsample #this is for fifteen minutes of simulates as the videos were recorded at 30Hz
env = MouseMEnv(render_mode='rgb_array')
# use a seed >200 to avoid initial states seen in the training dataset
env.seed(10000)

# get first observation, conditioned on threat level (TMT concentration)
obs, info = env.reset(food = 0, threat=0)

# keep a queue of last steps of observations
obs_deque = collections.deque(
    [obs] * obs_horizon, maxlen=obs_horizon)
# save visualization and rewards
imgs = [env.render()]
trajectory = [obs]
rewards = list()
done = False
step_idx = 0

with tqdm(total=max_steps, desc="Eval MouseMEnv") as pbar:
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
            
            
#plot some stats
points = np.vstack(trajectory)
print(np.sum(points[:,0])/len(points))

#set params
window_length = 60 #in nb of time steps (1Hz)
window_step = 60
T = len(points)
#compute mean with running window
phase = np.zeros(((T-window_length)//window_step,))
for j in range((T-window_length)//window_step):
    phase[j] = np.mean(points[j*window_step:j*window_step+window_length,0])
            
plt.figure()
plt.plot(phase)

#todo:
#write script that simulates many trials and loops over phases A, B & C
#save all those trajectories to compute stats to compare model to data
#answer questions




