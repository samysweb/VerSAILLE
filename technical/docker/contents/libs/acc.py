"""
Linear Adaptive Cruise Control in Relative Coordinates.
The visualization fixes the position of the leader car.
Adapation from N. Fulton and A. Platzer, "Safe Reinforcement Learning via Formal Methods: Toward Safe Control through Proof and Learning", AAAI 2018.
OpenAI Gym implementation adapted from the classic control cart pole environment.
"""

import logging
import math
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import random
import pygame
from pygame import gfxdraw

class ACCEnv(gym.Env):

    def __init__(self):
        self.MAX_VALUE = 100
        
        # Makes the continuous fragment of the system deterministic by fixing the
        # amount of time that the ODE evolves.
        self.TIME_STEP = 0.1

        # Maximal forward acceleration
        self.A = 3.1
        # Maximal braking acceleration
        self.B = 5.5

        
        bound = np.array([np.finfo(np.float32).max,np.finfo(np.float32).max])

        # Action Space: Choose Acceleration self.A, 0 or self.B
        self.action_space = spaces.Discrete(3)

        # State Space: (position, velocity)
        self.observation_space = spaces.Box(-bound, bound)

        self._seed()
        self.state = None

        # Rendering
        self.viewer = None

        self.render_mode="rgb_array"
        self.metadata = {
            'render_modes': ['rgb_array'],
            'video.frames_per_second' : 50
        }

    def is_crash(self, some_state):
      return some_state[0] <= 0

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%s (of type %s) invalid" % (str(action), type(action))
        
        # Get State
        state = self.state
        pos, vel = state[0],state[1]

        # Determine acceleration:
        acc = 0
        if action==0:
            # Relative Position -> Acceleration decreases relative distance
            acc = -self.A 
        elif action==1:
            acc = 0
        elif action==2:
            # Relative Position -> Braking increases relative distance
            acc = self.B
        else:
            raise ValueError(f"Unknown action value {action}")

        # update velocity by integrating the new acceleration over time --
        # pos = acc*t^2/2 + vel_0*t + pos_0
        # vel = vel = acc*t + vel_0
        t = self.TIME_STEP

        pos_0 = pos
        vel_0 = vel
        pos = acc*t**2/2 + vel_0*t + pos_0
        vel = acc*t + vel_0

        self.state = (pos, vel)

        crash = self.is_crash(self.state)
        truncated = self.state[0] > self.MAX_VALUE
        done = crash or truncated
        done = bool(done)

        if not done:
            # Well done, you stayed alive!
            reward = 0.1
        elif done and self.state[0] <= 1:
            # TOO CLOSE
            reward = -200.0
        elif done and self.state[0] > self.MAX_VALUE - 0.5:
            # Fell too far behind
            reward = -50.0
        else:
            assert False, "Not sure why this should happen, and when it was previously there was a bug in the if/elif guards..."
            reward = 0.0

        return np.array(self.state,dtype=np.float32), reward, done, truncated, {'crash': self.state[0] <= 0}

    def reset(self, seed=None,options=None):
        if seed is not None:
            self._seed(seed=seed)
        if options is not None and "new_state" in options:
            state = options["new_state"]
            assert (isinstance(state,list) or isinstance(state,tuple)) and len(state)==2, "New state must be tuple/list with 2 components"
            self.state = (np.float32(state[0]),np.float32(state[1]))
            return np.array(self.state), {'crash': self.is_crash(state)}
        pos = self.np_random.uniform(low=5, high=0.8*self.MAX_VALUE, size=(1,))[0]
        # We must not approach too fast (in which case braking would not stop us anymore)
        min_velocity = -np.sqrt(pos*2*self.B)
        # Hypothetical constraint on the other side:
        # (MAX_VALUE-pos) <= vel^2 / (2*B)
        # We must not fall behind too fast (in which case accelerating would not help us anymore)
        max_velocity = np.sqrt((self.MAX_VALUE-pos)*2*self.A)
        vel = self.np_random.uniform(low=min_velocity,high=max_velocity, size=(1,))[0]
        self.state = (np.float32(pos), np.float32(vel))

        return np.array(self.state), {'crash': False}

    def render(self, mode='rgb_array', close=False):
        assert mode==self.render_mode
        if close:
            if self.viewer is not None:    
                pygame.display.quit()
                pygame.quit()
                self.isopen = False
                self.viewer = None

        screen_width = 1000
        screen_height = 400

        carty = 200 # BOTTOM OF CART
        cartwidth = 150.0
        cartheight = 60.0
        x_scale = (screen_width-100-2*cartwidth)/self.MAX_VALUE

        relativeDistance = cartwidth * 2

        if self.viewer is None:
            pygame.init()
            self.viewer = pygame.Surface((screen_width, screen_height))

        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill((255, 255, 255))

        relativeDistance, relativeVelocity = self.state
        followerx = screen_width - 100 - relativeDistance*x_scale - cartwidth
        leaderx = screen_width - 100
                   
        # Add a follower cart.
        l,r = -cartwidth, 0.0
        t,b = cartheight, 0.0
        l += followerx
        r += followerx
        t += carty
        b += carty
        coords = [(l,b), (l,t), (r,t), (r,b)]
        gfxdraw.filled_polygon(self.surf, coords, (0,0,0))

        # Add leader cart
        l,r = -cartwidth, 0.0
        t,b = cartheight, 0.0
        l += leaderx
        r += leaderx
        t += carty
        b += carty
        coords = [(l,b), (l,t), (r,t), (r,b)]
        gfxdraw.filled_polygon(self.surf, coords, (0,0,0))

        # Display track
        gfxdraw.hline(self.surf, 0, screen_width, carty, (0, 0, 0))
        
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.viewer.blit(self.surf, (0, 0))
        return np.transpose(np.array(pygame.surfarray.pixels3d(self.viewer)), axes=(1, 0, 2))

    def close(self):
        if self.viewer is not None:    
            pygame.display.quit()
            pygame.quit()
            self.viewer = None


gym.register(
      id='acc-discrete-v0',
      entry_point=ACCEnv,
      max_episode_steps=410,  # todo edit
      reward_threshold=400.0, # todo edit
  )