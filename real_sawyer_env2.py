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


class RealSawyerBallBasketEnv(Env):
    metadata = {'render.modes':['human']}

    def __init__(self, driver, random_init=True):
        # raise NotImplementedError()
        super().__init__(
            config = '/home/robot/perls2/demos/reaching_config.yaml'
        )

        self.action_dim = 4 # action = [dx, dy]
        self.obs_dim = 4 # observation  = [eef_x, eef_y, eef_z, gripper]

        # define action space
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
        self.max_steps = 125
        self.steps = 0 

        # scaling factor from action -> osc control command
        self.ctrl_scale = 0.3

        # world origin (table center)
        self.origin = np.array([0.7075, 0.150, 0]) # TODO - Tune this: x,y = 0 at center of hoop. z=0 should be gripper barely touching table

        """
        TODO
        Define:
            - workspace boundaries
            - no entry zone bondaries (below hoop)
            - region above hoop
        """

        # workspace boundaries (low, high) - TODO - add z boundary and tune
        self.workspace_xlim = np.array([0.319, 0.775]) - self.origin[0] 
        self.workspace_ylim = np.array([-0.160, 0.467]) - self.origin[1]
        self.workspace_zlim = np.array([0, 0])- self.origin[2] # TODO - z-boundary

        # no entry zone - TODO: tune this
        self.noentry_xlim = np.array([0, 0])
        self.noentry_ylim = np.array([0, 0])
        self.noentry_zlim = np.array([0, 0])

        # region above hoop - TODO: tune this
        self.hoop_xlim = np.array([0, 0])
        self.hoop_ylim = np.array([0, 0])
        self.hoop_zlim = 0


    def reward(self): # TODO
        
        # chack ball is released (gripper is opened) above hoop
        eef_xpos = self.get_state()
        x_in_target = self.target_x[0] < eef_xpos[0] < self.target_x[1]
        y_in_target = self.target_y[0] < eef_xpos[1] < self.target_y[1]

        return int(x_in_target and y_in_target)


    def get_state(self): # TODO

        return self.driver.get_eef_xy() - self.origin


    def step(self, action, boundary=True):

        # scale input to controller
        # action = self.ctrl_scale * action # for 2d action space model
        sf = 1.5 # safety factor for boundary limit
        action = self.ctrl_scale * action[:3] # for 4d action space model

        # check workspace boundary condition
        eef_xpos = self.get_state()
        # print(f'Current position = {str(eef_xpos)}')
        x_in_bounds = self.x_lim[0] < eef_xpos[0] + action[0] < self.x_lim[1]
        y_in_bounds = self.y_lim[0] < eef_xpos[1] + action[1] < self.y_lim[1]
        if boundary:
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
            config = '/home/robot/perls2/demos/reaching_config.yaml',
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
        # self.max_steps = 500
        self.max_steps = 125
        self.steps = 0 

        # scaling factor from action -> osc control command
        # self.ctrl_scale = 0.075
        self.ctrl_scale = 0.3

        # world origin (table center)
        self.origin = np.array([0.7075, 0.150])

        # workspace boundaries (low, high)
        # x bondary: 0.900 0.832 0.345 0.319
        # y boundary: -0.171 0.486 -0.178 0.467 
        self.x_lim = np.array([0.43, 0.775]) - self.origin[0]
        self.y_lim = np.array([-0.160, 0.455]) - self.origin[1]
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

        # record previous action - for interpolation
        self.prev_action = np.zeros(2)


    def reward(self):
        
        eef_xpos = self.get_state()
        x_in_target = self.target_x[0] < eef_xpos[0] < self.target_x[1]
        y_in_target = self.target_y[0] < eef_xpos[1] < self.target_y[1]

        if x_in_target and y_in_target:
            return 100

        return 0
        # return int(x_in_target and y_in_target)


    def get_state(self):

        return self.driver.get_eef_xy() - self.origin


    def step(self, action, boundary=True):

        # scale input to controller
        # action = self.ctrl_scale * action # for 2d action space model
        sf = 1.2 # safety factor for boundary limit
        cur_action_weight = 1.0 # how much weight current action carries in interpolation

        # action_scaled = sf * self.ctrl_scale * action[:2] # for 4d action space model
        new_action = self.ctrl_scale * (cur_action_weight * action[:2] + (1 - cur_action_weight) * self.prev_action) / 2 # interpolation

        # check workspace boundary condition
        eef_xpos = self.get_state()
        # print(f'Current position = {str(eef_xpos)}')
        x_in_bounds = self.x_lim[0] < eef_xpos[0] + sf * new_action[0] < self.x_lim[1]
        y_in_bounds = self.y_lim[0] < eef_xpos[1] + sf * new_action[1] < self.y_lim[1]
        
        out_of_bounds = not x_in_bounds or not y_in_bounds

        if boundary:
            if out_of_bounds:
            # if not x_in_bounds or not y_in_bounds:
                # if next action will send eef out of boundary, ignore the action
                print("action out of bounds")
                new_action = np.zeros(self.action_dim)
                

        self.driver.step_axis(new_action)
        # self.driver.step_axis(self.ctrl_scale * action) # take action - no interpolation
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

        info = {'out_of_bounds': out_of_bounds} 

        self.steps += 1
        
        self.prev_action = action[:2]

        return observation, reward, done, info

    def _step_to_home(self, action):
        self.driver.step_axis(self.ctrl_scale * action)

    def _move_to_initial_pos(self):

        thresh = 0.01
        print(self.init_state - self.get_state())

        vec = self.init_state - self.get_state()
        while (abs(vec[0]) > thresh) or (abs(vec[1]) > thresh):
            action = self.init_state - self.get_state() # vector from current to initial position
            # print("-----RESET ACTION", action)
            self._step_to_home(action) # ignore boundary when reseting to make sure robo can return to home position
            vec = self.init_state - self.get_state()

        self.step(np.zeros(2))

    def _near_joint_limits(self, safety_factor=1.0):
        raise NotImplementedError
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
        time.sleep(3)
        self.driver.connect_and_reset_robot()
        print("-----Moved to initial pos---------")


        observation = self.get_state() # update observation
        # print("Observation", observation)

        if self.random_init:
            action_x = np.random.uniform(0, 0.15, 1)
            action_y = np.random.uniform(-0.25, 0.25, 1)
            action = np.concatenate([action_x, action_y, np.zeros(2)])
            print(f"-----Taking Random Action {action}------")
            for i in range(7):
                observation, _, _, _ = self.step(action)
        self.steps = 0

        print("--------------Finished Resetting. Starting in 2 sec-----------")
        time.sleep(2)

        return observation

    def render(self, mode='human'):
        return

    def close(self):
        self.close()

