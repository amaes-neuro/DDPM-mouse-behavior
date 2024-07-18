# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 14:32:13 2024

Custom environment for 2D mouse trajectories. It is simple: an action is a 2D step and added to current location.
Hunger is a function of distance to food, and threat is fixed for the entire episode.

@author: ahm8208
"""

import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces    
from matplotlib.path import Path
import shapely
import shapely.ops

class Mouse2DEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, size=5, reset_to_state=None):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self._seed = None
        self.seed()

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Box(low= np.concatenate( (np.array([-1, -25]) , np.zeros((18,)) , np.zeros((7,))) ), 
                                            high= np.concatenate( (np.array([86, 15]), 150*np.ones((18,)) , 3200*np.ones((7,))) ), 
                                            shape=(27,), 
                                            dtype=np.float64)


        # We have one continuous action (sign determines left or right)
        self.action_space = spaces.Box(low=np.array([-30,-25]), 
                                       high=np.array([35,25]), 
                                       shape=(2,), 
                                       dtype=np.float64)

        self.subsample = 10 #this can change, maybe do not hardcode
        #average points that make up the corners of the box in the dataset
        self.walls = [(-0.90,13.0),(35.53,14.88),(34.11,1.88),(51.00,1.68),(50.03,14.94),(85.83,12.31),
                      (85.83,-22.69),(49.54,-24.40),(50.23,-11.48),(35.01,-10.94),(35.23,-24.07),(-0.66,-21.24)] #this is static, the box does not change over time

        self.closed_box = self.walls
        self.closed_box.append( (-0.90,13.0) ) #make sure the box is closed
        self.closed_box = shapely.LineString( self.closed_box )

        self.box_path = Path( np.array(self.walls) )

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
        return np.hstack((self._agent_location,self._agent_sensory_field,self._agent_direction,
                          self._agent_side,self._agent_time,self._agent_food,self._agent_threat))
    

    def reset(self,location=None, direction= np.array([0,0]), side= np.array([0,0]), 
              time=0, food=0, threat=0, seed=None, options=None):    
        if location is not None:
            self._agent_location = location
        else:
            # Choose the agent's location uniformly at random
            self._agent_location = self.sample_initial_location()
        # Compute sensory field
        self._agent_sensory_field = self.compute_sensory_field(self._agent_location)
        #reset direction
        self._agent_direction = direction
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
        new_loc = self._agent_location + action #+ np.array([action[0]*np.random.randn()/10,action[1]*np.random.randn()/10])#i try to add noise to see if you get out of the stereotyped trajectories
        if self.box_path.contains_point( new_loc ):
            self._agent_location = new_loc
        else:     
            projection = shapely.ops.nearest_points(self.closed_box,shapely.Point((new_loc[0],new_loc[1])))
            self._agent_location = np.array([projection[0].x,projection[0].y])
            
        self._agent_sensory_field = self.compute_sensory_field(self._agent_location)

        if action[0]<0:
            self._agent_direction[0] = self._agent_direction[0] + 1
            self._agent_direction[1] = 0
        else:
            self._agent_direction[1] = self._agent_direction[1] + 1
            self._agent_direction[0] = 0

        if self._agent_location[0]<40:
            self._agent_side[0] = self._agent_side[0] + 1
            self._agent_side[1] = 0
        else:
            self._agent_side[1] = self._agent_side[1] + 1
            self._agent_side[0] = 0

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


    def sample_initial_location(self):
        bbox = [np.array([min(np.array(self.walls)[:,0]),min(np.array(self.walls)[:,1])]),np.array([max(np.array(self.walls)[:,0]),max(np.array(self.walls)[:,1])])]
        #Draw a random point in the bounding box of the convex hull
        rand_point = np.array([np.random.uniform(bbox[0][0], bbox[1][0]), np.random.uniform(bbox[0][1], bbox[1][1])])
        #We check if the random point is inside the convex hull, otherwise we draw it again            
        while self.box_path.contains_point(rand_point) == False:
            rand_point = np.array([np.random.uniform(bbox[0][0], bbox[1][0]), np.random.uniform(bbox[0][1], bbox[1][1])])
        return rand_point
        
        
    
    def compute_sensory_field(self, location):
        nb_angles = 18
        sensory_field = np.zeros((nb_angles,))
        for j in range(nb_angles):
            line = shapely.LineString([ (location[0],location[1]),
                                       (location[0]+1e3*np.cos(j*2*np.pi/nb_angles),location[1]+1e3*np.sin(j*2*np.pi/nb_angles)) ])
            inters = shapely.intersection(line, self.closed_box)
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
        

    def coord(self, list_of_coords):
        "Convert world coordinates to pixel coordinates."
        return [(10+sub[0], 30+sub[1]) for sub in list_of_coords] 
   
    
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
    
        # we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location[0]+10, self._agent_location[1]+30),
            3 #radius
            )
        
        # we draw the walls of the box
        pygame.draw.lines(
            canvas,
            (0,0,0),
            True, #connect last point to first point
            self.coord(self.walls),
            2 #thickness
            )
    
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
    