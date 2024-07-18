# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 17:17:34 2024

Custom environment for macroscopic mouse trajectories. It is simple: an action determines the side of the cage.

@author: ahm8208
"""


import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces    
import shapely
import shapely.ops

class MouseMEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, size=5, reset_to_state=None):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self._seed = None
        self.seed()

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Box(low= np.array([0,0,0,0,0]), 
                                            high= np.array([1,1200,1200,1,1]), 
                                            shape=(5,), 
                                            dtype=np.float64)


        # We have one continuous action (sign determines left or right)
        self.action_space = spaces.Box(low=np.array([0]), 
                                       high=np.array([1]), 
                                       shape=(1,), 
                                       dtype=np.float64)

        self.subsample = 30 #this can change, maybe do not hardcode

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None


    def _get_info(self):
        return {} #placeholder, not using for now
    

    def _get_obs(self):
        return np.hstack((self._agent_location,self._agent_side,self._agent_time,self._agent_food,self._agent_threat))
    

    def reset(self,location=0, side=0, time=0, food=0, threat=0, seed=None, options=None):    
        self._agent_location = location
        #reset sides
        self._agent_side = side
        #reset time in box
        self._agent_time = time
        #reset hunger variable
        self._agent_food = food
        #reset threat variable
        self._agent_threat = threat
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
    
        return observation, info
    
    
    def step(self, action):
        if action<0.5:
            action = 0
        else:
            action = 1
        
        if self._agent_location != action:
            self._agent_side = 0
            self._agent_location = action
        else:
            self._agent_side = self._agent_side + 1

        self._agent_time = self._agent_time + 1
        
        #food present and threat are constants        
        
        terminated = False # The behavior 'never' ends as we are interested in IRL
        reward = 0  
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
    
        return observation, reward, terminated, False, info
    
    
    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0,25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

           
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
    
    
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
    
        canvas = pygame.Surface((110, 60))
        canvas.fill((255, 255, 255))
#        pix_square_size = (
#            self.window_size / self.size
#        )  # The size of a single grid square in pixels
    
    
# =============================================================================
#         # Finally, add some gridlines
#         for x in range(self.size + 1):
#             pygame.draw.line(
#                 canvas,
#                 0,
#                 (0, pix_square_size * x),
#                 (self.window_size, pix_square_size * x),
#                 width=3,
#             )
#             pygame.draw.line(
#                 canvas,
#                 0,
#                 (pix_square_size * x, 0),
#                 (pix_square_size * x, self.window_size),
#                 width=3,
#             )    
# =============================================================================
        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
    
            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )    
    
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()    
    