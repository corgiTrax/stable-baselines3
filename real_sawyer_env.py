import gym
from gym import spaces
import numpy as np
import argparse
import sys
import torch
import time

import pickle5 as pickle


# stable baselines modules
# sys.path.insert(1, '/home/robot/perls2')
import stable_baselines3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.active_tamer.sac import SAC

# perls2 modules
from demos.sawyer_osc_2d import OpSpaceLineXYZ


class RealSawyerReachingEnv(gym.Env):
    metadata = {'render.modes':['human']}

    def __init__(self, driver):
        super(RealSawyerReachingEnv, self).__init__()

        self.action_dim = 4 # action = [dx, dy]
        self.obs_dim = 2 # observation  = [eef_x, eef_y]

        # define action space
        # self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(self.action_dim, ))#, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-0.25, -0.25, -0.25, -1.0]),
            high=np.array([0.25, 0.25, 0.25, 1.0]),
            shape=(4,),
            dtype=np.float32)

        # define observation space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim, ))#, dtype=np.float32)

        # define driver - perls2 controller, rewards, etc.
        self.driver = driver

        # max time steps
        self.max_steps = 900
        self.steps = 0 

        # scaling factor from action -> osc control command
        self.ctrl_scale = 0.075

        # world origin (table center)
        self.origin = np.array([0.7075, 0.150])
        # self.origin = np.array([0,0])


        # workspace boundaries (low, high)
        # x bondary: 0.900 0.832 0.345 0.319
        # y boundary: -0.171 0.486 -0.178 0.467 
        self.x_lim = np.array([0.319, 0.832]) - self.origin 
        self.y_lim = np.array([-0.178, 0.467]) - self.origin

        # Target boundaries
        # x boundary: 0.671 0.748 0.746 0.667
        # y boundary: 0.276 0.272 0.040 0.046
        self.target_x = np.array([0.671, 0.746]) - self.origin
        self.target_y =  np.array([0.046, 0.272]) - self.origin

    def reward(self):
        
        eef_xpos = self.get_state()
        x_in_target = self.target_x[0] < eef_xpos[0] < self.target_x[1]
        y_in_target = self.target_y[0] < eef_xpos[1] < self.target_y[1]

        return int(x_in_target and y_in_target)


    def get_state(self):

        return self.driver.get_eef_xy() - self.origin


    def step(self, action):

        # scale input to controller
        # action = self.ctrl_scale * action # for 2d action space model
        sf = 1.5 # safety factor for boundary limit
        action = self.ctrl_scale * action[:2] # for 4d action space model

        # check workspace boundary condition
        eef_xpos = self.get_state()
        print(f'Current position = {str(eef_xpos)}')
        x_in_bounds = self.x_lim[0] < eef_xpos[0] + action[0] < self.x_lim[1]
        y_in_bounds = self.y_lim[0] < eef_xpos[1] + action[1] < self.y_lim[1]
        
        if not x_in_bounds or not y_in_bounds:
            # if next action will send eef out of boundary, ignore the action
            print("action out of bounds")
            action = np.zeros(self.action_dim)

        self.driver.step_axis(action) # take action
        observation = self.get_state() # observation = [eef_x, eef_y]
        reward = self.reward()
        
        # done if:
        done = False

        if reward > 0:
            # task is completed
            done = True

        if self.steps > self.max_steps:
            # max steps is reached
            done = self.steps > self.max_steps # finish if reached maximum time steps
        
        info = {} 

        self.steps += 1

        return observation, reward, done, info

    
    def reset(self):

        print("----------------Resetting-----------------")
        self.driver.connect_and_reset_robot() # move to home position?

        observation = self.get_state() # update observation

        # random eef position intialization
        for i in range(50):
            action = np.random.uniform(-0.25, 0.25, 4)
            observation, _, _, _ = self.step(action)
        
        self.steps = 0

        print("--------------Finished Resetting. Starting in 5 sec-----------")
        time.sleep(5)

        return observation

    def render(self, mode='human'):
        return

    def close(self):
        self.close()
