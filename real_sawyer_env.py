import gym
from gym import spaces
import numpy as np
import argparse
import sys
import torch
import time
from playsound import playsound

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

    def __init__(
        self,driver,
        random_init=True,
        # terminate_on_ball_drop=True,
        max_steps=50,
        pickup_ball=False,
        ):

        super().__init__(
            config = '/home/robot/perls2/demos/reaching_config.yaml',
        )

        self.action_dim = 4 # action = [dx, dy]
        self.obs_dim = 4 # observation  = [eef_x, eef_y, eef_z, gripper]
        self.random_init = random_init

        # define action space
        self.action_space = spaces.Box(
            # low=np.array([-0.25, -0.25, -0.25, -1.0]),
            # high=np.array([0.25, 0.25, 0.25, 1.0]),
            low=np.array([-1.0, -1.0, -2.0, -1.0]),
            # low=np.array([-1.0, -1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 2.0, 1.0]),
            # high=np.array([1.0, 1.0, 1.0, 1.0]),
    
            shape=(4,),
            dtype=np.float32)

        # define observation space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim, ))#, dtype=np.float32)

        # define driver - perls2 controller, rewards, etc.
        self.driver = driver

        # max time steps
        # self.max_steps = 125
        self.max_steps = max_steps
        self.steps = 0 

        # scaling factor from action -> osc control command
        # self.ctrl_scale = 0.2
        self.ctrl_scale = 0.04
        # self.ctrl_scale = 0.1

        # world origin (table center) = raw [0.7003086084665489, 0.15554348938582682, -0.016173555136203836]
        self.origin = np.array([0.7003, 0.1555, -0.0162]) # x,y = 0 at center of hoop. z=0 is gripper barely touching table

        # gripper state
        self.gripper_state = -1 # -1 is closed, 1 is opened

        # initial position (in world frame)
        self.init_state = np.array([-0.25, 0, 0.233, 1])

        # terminate when ball is dropped outside the hoop?
        # self.terminate_on_ball_drop = terminate_on_ball_drop

        # pick up ball on start
        self.pickup_ball = pickup_ball

        """
        Define:
            - workspace boundaries
            - no entry zone bondaries (below hoop)
            - region above hoop
        """

        # workspace boundaries (low, high)
        """
        x low [0.43, 0.15554348938582682, -0.016173555136203836]
        x high 0.8058366143576214, 0.15841321512376866, 0.2299309528527574
        y low [0.44529973386747873, -0.10481875669917082, 0.22832600951498777]
        y high [0.44272720463759024, 0.38856754287548845, 0.22505078972861686]
        z low [0.4419352892760984, 0.1559964597727427, 0.009233602456942686]
        z high [0.799204655357289, 0.1540376849558416, 0.40385884829282903]
        self.x_lim = np.array([0.43, 0.775]) - self.origin[0]
        self.y_lim = np.array([-0.160, 0.455]) - self.origin[1]
        """

        self.workspace_xlim = np.array([0.43, 0.800]) - self.origin[0]
        self.workspace_ylim = np.array([-0.155, 0.450]) - self.origin[1]
        self.workspace_zlim = np.array([0.01, 0.40])- self.origin[2]

        # no entry zone
        """
        x lim [0.5370605937175418, 0.15555865122609916, 0.1129649458562731]
        y high [0.5954656575571576, 0.31905046891383176, 0.18542625239142904]
        y low [0.49131893716560815, -0.023878151184262852, 0.18189509401105028]
        z lim [0.588691088053821, 0.1550871075088405, 0.20496534744295652]
        """
        self.noentry_xlim = 0.53 - self.origin[0] # lower limit (upper limit is workspace limit)
        self.noentry_ylim = np.array([-0.024, 0.315]) - self.origin[1]
        self.noentry_zlim = 0.22 - self.origin[2] # lower limit (upper limit is workspace limit)

        # region above hoop
        """
        x low [0.6185053966606245, 0.1547914083598355, 0.23072427334606913]
        x high [0.7664866684, 0.23135682777, 0.230357174]
        y low [0.7809624586, 0.0889054314911561, 0.23799245666]
        y high [0.60075, 0.230271532, 0.2329407]
        """

        # self.hoop_xlim = np.array([0.620, 0.765]) - self.origin[0]
        self.hoop_xlim = np.array([0.604, 0.780]) - self.origin[0]
        # self.hoop_ylim = np.array([0.0890, 0.229]) - self.origin[1]
        self.hoop_ylim = np.array([-0.1, 0.1])
        self.hoop_zlim = 0.21 - self.origin[2] # lower limit 


    def reward(self):
        
        # check ball is released (gripper is opened) above hoop
        if self._check_is_above_hoop() and self.gripper_state == -1:
            return 100

        # elif self.gripper_state == 1 and not self._check_is_above_hoop():
        #     return 10

        elif not self._check_is_above_hoop() and self.gripper_state == -1:
            return -10

        return 0

    def get_state(self):

        # Returns [eef_x, eef_y, eef_z, gripper_state]

        return np.append(self.driver.get_eef_xyz() - self.origin, self.gripper_state)

    def _check_is_above_hoop(self):

        # Returns true if eef position is above hoop
        
        eef_xpos = self.get_state()[:3]
        return (
            self.hoop_xlim[0] < eef_xpos[0] < self.hoop_xlim[1]
            and self.hoop_ylim[0] < eef_xpos[1] < self.hoop_ylim[1]
            and eef_xpos[2] > self.hoop_zlim
        )

    def step(self, action):
        print("action", action)
        # scale input to controller
        sf_boundary = 1.3 # safety factor for boundary limit
        sf_noentry = 1.15 # safety factor for noentry zone
        cur_action_weight = 1.0 # how much weight current action carries in interpolation (1 means no interpolation)

        # new_action = self.ctrl_scale * (cur_action_weight * action[:3] + (1 - cur_action_weight) * self.prev_action) / 2 # interpolation
        new_action = np.copy(action)
        # print("action",action)
        new_action[:3] *= self.ctrl_scale

        # check workspace boundary condition
        eef_xpos = self.get_state()[:3]
        # print("eef_xpos", eef_xpos) #############################
        # print("new action", new_action)
        x_in_bounds = self.workspace_xlim[0] < eef_xpos[0] + sf_boundary * new_action[0] < self.workspace_xlim[1]
        y_in_bounds = self.workspace_ylim[0] < eef_xpos[1] + sf_boundary * new_action[1] < self.workspace_ylim[1]
        z_in_bounds = self.workspace_zlim[0] < eef_xpos[2] + sf_boundary * new_action[2] < self.workspace_zlim[1]
        
        out_of_bounds = not x_in_bounds or not y_in_bounds or not z_in_bounds

        # check no entry zone
        x_in_noentry =  self.noentry_xlim < eef_xpos[0] + sf_noentry * new_action[0]
        y_in_noentry = self.noentry_ylim[0] < eef_xpos[1] + sf_noentry * new_action[1] < self.noentry_ylim[1]
        z_in_noentry = eef_xpos[2] + sf_noentry * new_action[2] < self.noentry_zlim

        in_no_entry = x_in_noentry and y_in_noentry and z_in_noentry

        if out_of_bounds:
            # if next action will send eef out of boundary, ignore the action
            print(f"action out of bounds: {x_in_bounds} {y_in_bounds} {z_in_bounds}")
            new_action = np.zeros(self.action_dim)

        elif in_no_entry:
            # or if next action will send eef into no entry zone, ignore the action
            print(f"not allowed in this area: {x_in_noentry} {y_in_noentry} {z_in_noentry}")
            new_action = np.zeros(self.action_dim)

        # take gripper action
        if new_action[-1] >= 0:
            # gripper should be closed
            if self.gripper_state == -1:
                # if gripper is currently open
                self.robot_interface.close_gripper()
                self.gripper_state = 1
        else:
            # gripper should be opened
            if self.gripper_state == 1:
                # if gripper is currently closed
                self.robot_interface.open_gripper()
                self.gripper_state = -1

        # take action in cartesian space
        self.driver.step_axis(new_action)

        observation = self.get_state() # observation = [eef_x, eef_y, eef_z, gripper]
        reward = self.reward()
        
        # done if:
        done = False

        if reward > 0:
            # task is completed
            print("---------COMPLETED TASK!-----------")
            # playsound("success.wav", block=False)
            done = True

        if self.steps > self.max_steps:
            # max steps is reached
            done = True # finish if reached maximum time steps

        if self.gripper_state == -1:
            print("Gripper opened")
            if not self._check_is_above_hoop():
                # ball got dropped outside the hoop
                print("--------DROPPED BALL...----------")
                # playsound("fail.wav", block=False)
                done = True

        info = {}
        info['out_of_bounds'] = out_of_bounds
        info['in_noentry'] = in_no_entry
        info['above_hoop'] = self._check_is_above_hoop()

        self.steps += 1

        if self._check_is_above_hoop(): ########################
            print("========================ABOVE HOOP=============================")
        
        return observation, reward, done, info
    
    def _step_to_home(self, action):
        # self.driver.step_axis(self.ctrl_scale * action)
        self.driver.step_axis(self.ctrl_scale * 4 * action)


    def _move_to_initial_pos(self):

        thresh = 0.01
        # print(self.init_state - self.get_state()[:3])

        # if in position with possible collision, first move up
        if self.get_state()[0] > self.noentry_xlim:
            # print("WAYPOINT RESET")
            waypoint = self.get_state()
            waypoint[2] = 0.235
            vec = waypoint - self.get_state()
            while abs(vec[2] > thresh):
                print(vec)
                self._step_to_home(vec[:3])# + np.random.uniform(-0.2, 0.2, 3))
                vec = waypoint - self.get_state()

        # move to home position
        vec = self.init_state - self.get_state()
        while (abs(vec[0]) > thresh or abs(vec[1]) > thresh or abs(vec[2]) > thresh):
            # action = self.init_state - self.get_state()[:3] # vector from current to initial position
            self._step_to_home(vec[:3]) # ignore boundary when reseting to make sure robo can return to home position
            vec = self.init_state - self.get_state()

        self.step(np.zeros(4))

    def reset(self):

        print("----------------Resetting-----------------")
        self._move_to_initial_pos()
        time.sleep(2)
        self.driver.connect_and_reset_robot()
        print("-----Moved to initial pos---------")

        observation = self.get_state() # update observation

        if self.pickup_ball:
            for i in range(5): # move down
                action = np.array([0, 0, -0.3, -1])
                self._step_to_home(action)
            
            self.gripper_state = -1
            
            # grip ball
            self.robot_interface.close_gripper()
            self.gripper_state = 1

            self._move_to_initial_pos()

        else:
            # close gripper 
            self.robot_interface.close_gripper()
            self.gripper_state = 1

        # random initialization
        if self.random_init:
            # action_y = np.random.uniform(-0.15, 0.15, 1)
            # action_z = np.random.uniform(-0.1, 0.065, 1) 
            action_y = np.random.uniform(-0.75, 0.75, 1)
            # action_z = np.random.uniform(-0.5, 0.325, 1) 
            random_action = np.array([0.02, action_y[0], -0.5, 1]) # random initializaation in y only (fixed x-home and fixed z-below hoop)
            print("Random Init Action", random_action)
            for i in range(7):
                observation, _, _, _ = self.step(random_action)

        self.steps = 0

        print("--------------Finished Resetting. Starting in 1 sec-----------")
        time.sleep(1)

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
        self.max_steps = 250
        self.steps = 0 

        # scaling factor from action -> osc control command
        self.ctrl_scale = 0.2

        # world origin (table center)
        self.origin = np.array([0.7075, 0.150])

        # workspace boundaries (low, high)
        # x bondary: 0.900 0.832 0.345 0.319
        # y boundary: -0.171 0.486 -0.178 0.467 
        self.x_lim = np.array([0.43, 0.775]) - self.origin[0]
        self.y_lim = np.array([-0.160, 0.455]) - self.origin[1]

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
            self._step_to_home(action) # ignore boundary when reseting to make sure robo can return to home position
            vec = self.init_state - self.get_state()

        self.step(np.zeros(2))

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
            # action = np.concatenate([action_x, action_y, np.zeros(2)])
            action = np.concatenate([0, action_y, -0.3, 0])

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

