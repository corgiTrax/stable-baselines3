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
from perls2.envs.env import Env


class RealSawyerReachingEnv3d(gym.Env):
    metadata = {'render.modes':['human']}

    def __init__(self, driver):
        # raise NotImplementedError()
        super(RealSawyerReachingEnv3d, self).__init__()

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
        self.ctrl_scale = 0.04

        # world origin (table center)
        self.origin = np.array([0.7075, 0.150])
        # self.origin = np.array([0,0])


        # workspace boundaries (low, high)
        # x bondary: 0.900 0.832 0.345 0.319
        # y boundary: -0.171 0.486 -0.178 0.467 
        self.x_lim = np.array([0.319, 0.775]) - self.origin 
        self.y_lim = np.array([-0.160, 0.467]) - self.origin
        # TODO - z-boundary

        # TODO
        # Target boundaries
        # x boundary: 0.671 0.748 0.746 0.667
        # y boundary: 0.276 0.272 0.040 0.046
        self.target_x = np.array([0.671, 0.746]) - self.origin
        self.target_y =  np.array([0.046, 0.272]) - self.origin

    def reward(self): # TODO
        
        eef_xpos = self.get_state()
        x_in_target = self.target_x[0] < eef_xpos[0] < self.target_x[1]
        y_in_target = self.target_y[0] < eef_xpos[1] < self.target_y[1]

        return int(x_in_target and y_in_target)


    def get_state(self): # TODO

        return self.driver.get_eef_xy() - self.origin


    def step(self, action):

        # scale input to controller
        # action = self.ctrl_scale * action # for 2d action space model
        sf = 1.5 # safety factor for boundary limit
        action = self.ctrl_scale * action[:3] # for 4d action space model

        # check workspace boundary condition
        eef_xpos = self.get_state()
        # print(f'Current position = {str(eef_xpos)}')
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
            print("---------TARGET REACHED-----------")
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

class RealSawyerReachingEnv(Env):
    metadata = {'render.modes':['human']}

    def __init__(self, driver, random_init=True):
        super().__init__(
            config = '/home/robot/perls2/demos/demo_control_cfg.yaml',
        )

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
        self.ctrl_scale = 0.03

        # world origin (table center)
        self.origin = np.array([0.7075, 0.150])

        # workspace boundaries (low, high)
        # x bondary: 0.900 0.832 0.345 0.319
        # y boundary: -0.171 0.486 -0.178 0.467 
        self.x_lim = np.array([0.43, 0.775]) - self.origin[0]
        self.y_lim = np.array([-0.160, 0.467]) - self.origin[1]
        # self.x_lim = np.array([-0.30, 0.14])
        # self.y_lim = np.array([-0.30, 0.14])


        # Target boundaries
        # x boundary: 0.671 0.748 0.746 0.667
        # y boundary: 0.276 0.272 0.040 0.046
        self.target_x = np.array([0.671, 0.746]) - self.origin[0]
        self.target_y =  np.array([0.046, 0.272]) - self.origin[1]

        # Whether to use random position initialization
        self.random_init = random_init

        # Initial position
        self.init_state = np.array([-0.25, 0])

        # print("------NEUTRAL POS", self.robot_interface.limb_neutral_positions)
        # print("------NEUTRAL EEF POS", self.robot_interface.ee_position)
        # print("------NEUTRAL EEF ORI", self.robot_interface.ee_orientation)

        # print("--------INITIAL POS", self.get_state())
        # print("-----JOINTS", self.robot_interface.q)

        # get sawyer joint limits
        joint_limits = self.robot_interface.get_joint_limits()
        # print("JOINT LIMITS", joint_limits)
        joint_lim_lower, joint_lim_upper = [], []
        for i in range(7):
            joint = joint_limits[i]
            joint_lim_lower.append(joint['lower'])
            joint_lim_upper.append(joint['upper'])

        self.joint_lim_lower = np.array(joint_lim_lower)
        self.joint_lim_upper = np.array(joint_lim_upper)

        # print("LOWER LIMITS", self.joint_lim_lower)
        # print("UPPER LIMITS", self.joint_lim_upper)

        # record previous action
        self.prev_action = np.zeros(2)


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
        sf = 1.2 # safety factor for boundary limit
        action_scaled = sf * self.ctrl_scale * action[:2] # for 4d action space model

        # check workspace boundary condition
        eef_xpos = self.get_state()
        # print(f'Current position = {str(eef_xpos)}')
        x_in_bounds = self.x_lim[0] < eef_xpos[0] + action_scaled[0] < self.x_lim[1]
        y_in_bounds = self.y_lim[0] < eef_xpos[1] + action_scaled[1] < self.y_lim[1]
        
        if not x_in_bounds or not y_in_bounds:
            # if next action will send eef out of boundary, ignore the action
            print("action out of bounds")
            action = np.zeros(self.action_dim)

        new_action = self.ctrl_scale * (action[:2] + self.prev_action) / 2 # interpolation
        self.driver.step_axis(new_action)
        # print("deltas", self.ctrl_scale * action)
        # print("eef pos", self.get_state())
        # self.driver.step_axis(self.ctrl_scale * action) # take action
        observation = self.get_state() # observation = [eef_x, eef_y]
        reward = self.reward()
        
        # done if:
        done = False

        if reward > 0:
            # task is completed
            print("---------TARGET REACHED-----------")
            done = True

        if self.steps > self.max_steps:
            # max steps is reached
            done = self.steps > self.max_steps # finish if reached maximum time steps
        
        # if self._near_joint_limits(safety_factor=0.9):
        #     print("CLOSE TO JOINT LIMIT")

        info = {} 

        self.steps += 1
        # print("eef pos", observation)
        # print(f"xlim: {self.x_lim}, ylim: {self.y_lim}")
        
        self.prev_action = action[:2]

        return observation, reward, done, info

    def _move_to_initial_pos(self):

        thresh = 0.01
        print(self.init_state - self.get_state())

        vec = self.init_state - self.get_state()
        while (abs(vec[0]) > thresh) or (abs(vec[1]) > thresh):
            action = self.init_state - self.get_state() # vector from current to initial position
            # print("-----RESET ACTION", action)
            self.step(action)
            vec = self.init_state - self.get_state()

    def _near_joint_limits(self, safety_factor=1.0):
        # self.robot_interface.step()
        print("lower", self.joint_lim_lower)
        print("upper", self.joint_lim_upper)
        print("joints", self.robot_interface.q)


        return (
            np.any(self.robot_interface.q < safety_factor * self.joint_lim_lower)
            or np.any(self.robot_interface.q > safety_factor * self.joint_lim_upper)
            )
    
    def reset(self):

        print("----------------Resetting-----------------")

        self._move_to_initial_pos()
        print("-----Moved to initial pos---------")
        time.sleep(2)

        observation = self.get_state() # update observation
        # print("Observation", observation)

        # random eef position intialization
        if self.random_init:
            action_x = np.random.uniform(0, 0.035, 1)
            action_y = np.random.uniform(-0.070, 0.070, 1)
            action = np.concatenate([action_x, action_y, np.zeros(2)])
            print(f"-----Taking Random Action {action}------")
            for i in range(50):
                observation, _, _, _ = self.step(action)
        
        self.steps = 0

        print("--------------Finished Resetting. Starting in 3 sec-----------")
        time.sleep(3)

        return observation

    def render(self, mode='human'):
        return

    def close(self):
        self.close()


# class RealSawyerReachingEnv(gym.Env):
#     metadata = {'render.modes':['human']}

#     def __init__(self, driver, random_init=True):
#         super(RealSawyerReachingEnv, self).__init__()

#         self.action_dim = 4 # action = [dx, dy]
#         self.obs_dim = 2 # observation  = [eef_x, eef_y]

#         # define action space
#         # self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(self.action_dim, ))#, dtype=np.float32)
#         self.action_space = spaces.Box(
#             low=np.array([-0.25, -0.25, -0.25, -1.0]),
#             high=np.array([0.25, 0.25, 0.25, 1.0]),
#             shape=(4,),
#             dtype=np.float32)

#         # define observation space
#         self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim, ))#, dtype=np.float32)

#         # define driver - perls2 controller, rewards, etc.
#         self.driver = driver

#         # max time steps
#         self.max_steps = 900
#         self.steps = 0 

#         # scaling factor from action -> osc control command
#         self.ctrl_scale = 0.075

#         # world origin (table center)
#         self.origin = np.array([0.7075, 0.150])

#         # workspace boundaries (low, high)
#         # x bondary: 0.900 0.832 0.345 0.319
#         # y boundary: -0.171 0.486 -0.178 0.467 
#         self.x_lim = np.array([0.319, 0.775]) - self.origin 
#         self.y_lim = np.array([-0.160, 0.467]) - self.origin

#         # Target boundaries
#         # x boundary: 0.671 0.748 0.746 0.667
#         # y boundary: 0.276 0.272 0.040 0.046
#         self.target_x = np.array([0.671, 0.746]) - self.origin
#         self.target_y =  np.array([0.046, 0.272]) - self.origin

#         # Whether to use random position initialization
#         self.random_init = random_init



#     def reward(self):
        
#         eef_xpos = self.get_state()
#         x_in_target = self.target_x[0] < eef_xpos[0] < self.target_x[1]
#         y_in_target = self.target_y[0] < eef_xpos[1] < self.target_y[1]

#         return int(x_in_target and y_in_target)


#     def get_state(self):

#         return self.driver.get_eef_xy() - self.origin


#     def step(self, action):

#         # scale input to controller
#         # action = self.ctrl_scale * action # for 2d action space model
#         sf = 1.5 # safety factor for boundary limit
#         action = self.ctrl_scale * action[:2] # for 4d action space model

#         # check workspace boundary condition
#         eef_xpos = self.get_state()
#         # print(f'Current position = {str(eef_xpos)}')
#         x_in_bounds = self.x_lim[0] < eef_xpos[0] + action[0] < self.x_lim[1]
#         y_in_bounds = self.y_lim[0] < eef_xpos[1] + action[1] < self.y_lim[1]
        
#         if not x_in_bounds or not y_in_bounds:
#             # if next action will send eef out of boundary, ignore the action
#             print("action out of bounds")
#             action = np.zeros(self.action_dim)

#         self.driver.step_axis(action) # take action
#         observation = self.get_state() # observation = [eef_x, eef_y]
#         reward = self.reward()
        
#         # done if:
#         done = False

#         if reward > 0:
#             # task is completed
#             print("---------TARGET REACHED-----------")
#             done = True

#         if self.steps > self.max_steps:
#             # max steps is reached
#             done = self.steps > self.max_steps # finish if reached maximum time steps
        
#         info = {} 

#         self.steps += 1

#         return observation, reward, done, info

    
#     def reset(self):

#         print("----------------Resetting-----------------")
#         # self.driver.connect_and_reset_robot() # move to home position?


#         observation = self.get_state() # update observation
#         print("Observation", observation)

#         # random eef position intialization
#         if self.random_init:
#             action_x = np.random.uniform(-0.02, 0.04, 1)
#             action_y = np.random.uniform(-0.075, 0.075, 1)
#             action = np.concatenate([action_x, action_y, np.zeros(2)])
#             print(f"-----Action is {action}------")
#             for i in range(50):
#                 # action_x = np.random.uniform(-0.02, 0.04, 1)
#                 # action_y = np.random.uniform(-0.075, 0.075, 1)
#                 # action = np.concatenate([action_x, action_y, np.zeros(2)])
#                 # action = np.random.uniform(-0.25, 0.25, 4)
#                 observation, _, _, _ = self.step(action)
        
#         self.steps = 0

#         print("--------------Finished Resetting. Starting in 3 sec-----------")
#         time.sleep(3)

#         return observation

#     def render(self, mode='human'):
#         return

#     def close(self):
#         self.close()
