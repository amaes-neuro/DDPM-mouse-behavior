# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:52:53 2024

This script trains a denoising diffusion model.

1. Load dataset
2. Setup network model
3. Training loop

This code is adapted from https://github.com/real-stanford/diffusion_policy/tree/main

@author: ahm8208
"""

from MouseTrajectoryDataset_M import MouseTrajectoryDataset
from ConditionalDiffuser import ConditionalUnet1D
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import numpy as np
import pickle

"""
Load dataset
"""

# parameters
pred_horizon = 4
obs_horizon = 1
action_horizon = 1


# create dataset from file
dataset = MouseTrajectoryDataset(
    dataset_path='data/',
    pred_horizon=pred_horizon,
    obs_horizon=obs_horizon,
    action_horizon=action_horizon
)
# save training data statistics (min, max) for each dim
stats = dataset.stats

# create dataloader
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256,
    num_workers=0,
    shuffle=True,
)

# visualize data in batch
batch = next(iter(dataloader))
print("batch['obs'].shape:", batch['obs'].shape)
print("batch['action'].shape", batch['action'].shape)



"""
Setup model
"""

obs_dim = 6
action_dim = 1

# create network object
noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*obs_horizon
)

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

# device transfer
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
_ = noise_pred_net.to(device)



"""
Train model
"""

num_epochs = 50

# Exponential Moving Average
# accelerates training and improves stability
# holds a copy of the model weights
ema = EMAModel(
    parameters=noise_pred_net.parameters(),
    use_ema_warmup=True, #try this out
    power=0.8)

# Standard ADAM optimizer
# Note that EMA parametesr are not optimized
optimizer = torch.optim.AdamW(
    params=noise_pred_net.parameters(),
    lr=1e-4, weight_decay=1e-6)

# Cosine LR schedule with linear warmup
lr_scheduler = get_scheduler(
    name='cosine',
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(dataloader) * num_epochs
)

with tqdm(range(num_epochs), desc='Epoch') as tglobal:
    # epoch loop
    for epoch_idx in tglobal:
        epoch_loss = list()
        # batch loop
        with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
            for nbatch in tepoch:
                # data normalized in dataset
                # device transfer
                nobs = nbatch['obs'].to(device)
                naction = nbatch['action'].to(device)
                B = nobs.shape[0]

                # observation as FiLM conditioning
                # (B, obs_horizon, obs_dim)
                obs_cond = nobs[:,:obs_horizon,:]
                # (B, obs_horizon * obs_dim)
                obs_cond = obs_cond.flatten(start_dim=1)

                # I want to see if adding noise to obs_cond can help generalization
                #obs_cond = obs_cond + torch.randn(obs_cond.shape, device=device)/20

                # sample noise to add to actions
                noise = torch.randn(naction.shape, device=device)

                # sample a diffusion iteration for each data point
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (B,), device=device
                ).long()

                # add noise to the clean images according to the noise magnitude at each diffusion iteration
                # (this is the forward diffusion process)
                noisy_actions = noise_scheduler.add_noise(
                    naction, noise, timesteps)

                # predict the noise residual
                noise_pred = noise_pred_net(
                    noisy_actions, timesteps, global_cond=obs_cond)

                # L2 loss
                loss = torch.nn.functional.mse_loss(noise_pred, noise)

                # optimize
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # step lr scheduler every batch
                # this is different from standard pytorch behavior
                lr_scheduler.step()

                # update Exponential Moving Average of the model weights
                ema.step(noise_pred_net.parameters())

                # logging
                loss_cpu = loss.item()
                epoch_loss.append(loss_cpu)
                tepoch.set_postfix(loss=loss_cpu)
        tglobal.set_postfix(loss=np.mean(epoch_loss))
        with open('checkpoints/training_status.pickle', 'wb') as file:
            pickle.dump(epoch_idx, file)

        

# Weights of the EMA model
# is used for inference
ema_noise_pred_net = noise_pred_net
ema.copy_to(ema_noise_pred_net.parameters())

#save weights
path = 'checkpoints/t_M_15.pt'
torch.save(ema_noise_pred_net.state_dict(), path)
print('Model saved to'+path)


#Note that training does not interact with the environment.
#So I have to take care that the data is set up such that it is compatible with the environment.
#
#Question: should the dataset be normalized?

#t1: (4,1,4) 4 epochs
#t2: (4,1,4) 20 epochs
#t3: (4,1,1) 20 epochs

#t_M_1 (4,1,1) 20 epochs -> after generating synthetic data it seems to do very well on phase A, and then increasingly bad
#t_M_2 (4,1,1) 20 epochs, probability of 75% to skip a phase A training batch to balance the dataset -> loss 0.0452
#t_M_3 (4,1,1) 50 epochs, same as t_M_2 but more training -> loss 0.0443
#t_M_4 (4,1,1) 50 epochs, lr=1e-4 -> lr=5e-5 because 50 epochs does not do much better than 20 epochs -> loss 0.0448
#t_M_5 (4,1,1) 75 epochs, lr=1e-4 -> loss 0.0449
#t_M_6 (4,1,1) 75 epochs, lr=1e-4 and kernel_size = 3 instead of 5 -> loss 0.0423
#t_M_7 (8,1,1) 75 epochs, lr=1e-4, kernel_size=3 -> loss 0.0425
#t_M_8 (4,1,4) 75 epochs, lr=1e-4, kernel_size=3 -> loss similar

#t_M_9 (4,1,1) 75 epochs, added an additional 90C to data in attempt to give it more weight -> loss 0.0432
#it does seem that choosing the dataset is very important! i have to make a choice on how to balance the data, what matters?

#t_M_10 (4,1,1) 75 epochs, removed additional 90C mice, but recomputed threat state for phase C -> loss 0.044

#t_M_11 (4,1,1) 50 epochs, balanced4 dataset (additional state) -> loss 0.0436

#t_M_12 (4,1,1) 50 epochs, balanced5 dataset (naive threat state) -> loss 0.0446

#t_M_13 (4,1,1) 50 epochs, balanced6 dataset (behavioral encoded threat state) -> loss 0.0368
#t_M_14 (4,1,1) 100 epochs, " " " -> loss 0.0357

#t_M_15 (4,1,1) 50 epochs, balanced7 dataset (simple switching, different normalization) -> loss 0.042


