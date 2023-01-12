from __future__ import print_function
from __future__ import print_function

import itertools
import os
import random
import sys
import threading as thread
from typing import Callable
import argparse
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from lunar_lander_models import LunarLanderExtractor, LunarLanderStatePredictor
from PyQt5.QtWidgets import *
import math
import collections
import copy
import time
import pdb

from stable_baselines3.active_tamer.active_tamerRL_sac_record_ballbasket import ActiveTamerRLSACOptimBallBasket
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.sac.sac import SAC

import robosuite as suite
from robosuite import wrappers
from robosuite import load_controller_config

sys.path.insert(1, '/home/robot/perls2')

from demos.sawyer_osc_3d import OpSpaceLineXYZ
from real_sawyer_env import RealSawyerBallBasketEnv
import argparse

from stable_baselines3.common.human_feedback import HumanFeedback
from stable_baselines3.common.dummy_learning_interface import FeedbackInterface

class BallBasketSceneGraph:
    def __init__(self):
        self.agent = {'location': {'x': 0, 'y': 0, 'z': 0, 'g': 0}}
        self.num_feedback_given = collections.Counter()
        self.aRPE_average = collections.Counter()
        self.curr_graph = None
        self.total_feedback = 50000 #200000 for frequency based scene graph
        self.given_feedback = 0
        self.total_timesteps = 0
        self.model_training = 2 #changeUCB
    
    def calculate_ucb(self, graph):
        # pdb.set_trace()       
        return self.aRPE_average[tuple(graph)] + 2 * math.sqrt(2 * self.given_feedback / (self.num_feedback_given[tuple(graph)] + 1))       
    
    def getUCBRank(self):
        curr_graph_ucb1 = self.calculate_ucb(self.curr_graph)
        rank = 0
        for graph in self.num_feedback_given:
            print(graph, self.calculate_ucb(graph))
            if self.calculate_ucb(graph) > curr_graph_ucb1:
                rank += 1
        print(f'Rank = {str(rank)} Graph = {self.curr_graph}')
        return rank

    def right(self, obj_a):
        return obj_a['location']['x'] < 0

    # def center(self, obj_a):
    #     return obj_a['location']['x'] > -0.1 and obj_a['location']['x'] < 0.1

    def left(self, obj_a):
        return obj_a['location']['x'] > 0

    
    def top(self, obj_a):
        return obj_a['location']['y'] < 0

    # def middle(self, obj_a):
    #     return obj_a['location']['y'] > -0.1 and obj_a['location']['y'] < 0.1

    def bottom(self, obj_a):
        return obj_a['location']['y'] > 0
    
    
    def above(self, obj_a):
        return obj_a['location']['z'] > 0.226
    
    def below(self, obj_a):
        return obj_a['location']['z'] < 0.226

    
    def gripper_open(self, obj_a):
        return obj_a['location']['g'] < 0
    
    def gripper_close(self, obj_a):
        return obj_a['location']['g'] > 0

    def above_hoop(self, obj_a):
        return obj_a['location']['x'] > -0.1 and obj_a['location']['x'] < 0.1 \
                and obj_a['location']['y'] > -0.1 and obj_a['location']['y'] < 0.1 \
                and obj_a['location']['z'] > 0.226
    # add a state of gripper open/close
    

    def updateRPE(self, human_feedback, human_critic_prediction):
        
        self.num_feedback_given[tuple(self.curr_graph)] += 1

        self.given_feedback += 1
        # pdb.set_trace()
        self.aRPE_average[tuple(self.curr_graph)] *= (self.num_feedback_given[tuple(self.curr_graph)] - 1)/self.num_feedback_given[tuple(self.curr_graph)]
        self.aRPE_average[tuple(self.curr_graph)] += abs(human_feedback - human_critic_prediction)/self.num_feedback_given[tuple(self.curr_graph)]
        self.aRPE_average[tuple(self.curr_graph)] = self.aRPE_average[tuple(self.curr_graph)].detach()


    def getCurrGraph(self):
        self.curr_graph = [self.gripper_open(self.agent), self.gripper_close(self.agent), self.model_training]
        
        if self.model_training == 0:
            self.curr_graph.extend([self.right(self.agent), self.left(self.agent),])
        
        if self.model_training == 1:
            self.curr_graph.extend([self.top(self.agent), self.bottom(self.agent)])
        
        if self.model_training == 2:
            self.curr_graph.extend([self.above(self.agent), self.below(self.agent)])
        
        if self.model_training == 3:
            self.curr_graph.extend([self.above_hoop(self.agent)])
        
        return self.curr_graph
        
    def updateGraph(self, newState, action, model_training): #changeUCB
        # pdb.set_trace()
        prev_graph = copy.deepcopy(self.curr_graph)
        self.agent['location'] = {'x': newState[0][0], 'y': newState[0][1], 'z': newState[0][2], 'g': newState[0][3]}
        self.total_timesteps += 1
        self.model_training = model_training #changeUCB
        return self.getCurrGraph() != prev_graph, self.getUCBRank() <= 3 #changeUCB



def train_model(model, config_data, feedback_gui, human_feedback, env):
    model.learn(
        config_data["steps"],
        human_feedback_gui=feedback_gui,
        human_feedback=human_feedback,
        reset_num_timesteps = False
    )
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=20, render=True
    )
    print(f"After Training: Mean reward: {mean_reward} +/- {std_reward:.2f}")


def main():
    
    with open("configs/real_robot/active_tamer_rl_sac_split.yaml", "r") as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)

    # if args.seed:
    #     config_data['seed'] = args.seed

    tensorboard_log_dir = config_data["tensorboard_log_dir"]

    parser = argparse.ArgumentParser(
        description="Test controllers and measure errors.")
    parser.add_argument('--world', default='Real', help='World type for the demo, uses config file if not specified', choices=['Bullet', 'Real'])
    parser.add_argument('--robot', default='sawyer', help='Robot type overrides config', choices=['panda', 'sawyer'])
    parser.add_argument('--ctrl_type',
                        default="EEImpedance",
                        help='Type of controller to test')
    parser.add_argument('--demo_type',
                        default="Line",
                        help='Type of menu to run.')
    parser.add_argument('--test_fn',
                        default='set_ee_pose',
                        help='Function to test',
                        choices=['set_ee_pose', 'move_ee_delta', 'set_joint_delta', 'set_joint_positions', 'set_joint_torques', 'set_joint_velocities'])
    parser.add_argument('--path_length', type=float,
                        default=None, help='length in m of path')
    parser.add_argument('--delta_val',
                        default=[0.001, 0.001, 0.001], type=float,
                        help="Max step size (m or rad) to take for demo.")
    parser.add_argument('--axis',
                        default='x', type=str,
                        choices=['x', 'y', 'z'],
                        help='axis for demo. Position direction for Line or rotation axis for Rotation')
    parser.add_argument('--num_steps', default=1, type=int,
                        help="max steps for demo.")
    parser.add_argument('--plot_pos', action="store_true",
                        help="whether to plot positions of demo.")
    parser.add_argument('--plot_error', action="store_true",
                        help="whether to plot errors.")
    parser.add_argument('--save', action="store_true",
                        help="whether to store data to file")
    parser.add_argument('--demo_name', default=None,
                        type=str, help="Valid filename for demo.")
    parser.add_argument('--save_fig', action="store_true",
                        help="whether to save pngs of plots")
    parser.add_argument('--fix_ori', action="store_true", default=True,
                        help="fix orientation for move_ee_delta")
    parser.add_argument('--fix_pos', action="store_true",
                        help="fix position for move_ee_delta")
    parser.add_argument('--config_file', default='/home/robot/perls2/demos/demo_control_cfg.yaml', help='absolute filepath for config file.')
    parser.add_argument('--cycles', type=int, default=1, help="num times to cycle path (only for square)")
    args = parser.parse_args()
    kwargs = vars(args)

    driver = OpSpaceLineXYZ(**kwargs)

    # env = RealSawyerBallBasketEnv(driver, random_init=False)
    env = RealSawyerBallBasketEnv(driver, random_init=True)

    time.sleep(1)
    human_feedback = HumanFeedback()
    feedback_gui = FeedbackInterface()

    np.set_printoptions(threshold=np.inf)

    policy_kwargs = dict(
        net_arch=[32, 32],
    )
    os.makedirs(tensorboard_log_dir, exist_ok=True)

    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8
    kwargs = dict(seed=0)
    kwargs.update(dict(buffer_size=1))

    custom_objects = {}
    if newer_python_version:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }
    # trained_model = SAC.load(
    #     config_data["trained_model"], env, custom_objects=custom_objects, **kwargs
    # )

    while os.path.exists(config_data['human_data_save_path']):
        config_data['human_data_save_path'] = "/".join(config_data['human_data_save_path'].split("/")[:-1]) + '/participant_' + str(int(random.random() * 1000000000))

    # scene_graph_instances = [BallBasketSceneGraph() for _ in range(4)]
    model = ActiveTamerRLSACOptimBallBasket( # TODO - replace this model with real robot equivalent of ActiveTamerRLSACOptimBallBasket
        config_data["policy_name"],
        env,
        verbose=config_data["verbose"],
        tensorboard_log=tensorboard_log_dir,
        policy_kwargs=policy_kwargs,
        save_every=config_data["save_every_steps"],
        learning_rate=config_data["learning_rate"],
        buffer_size=config_data["buffer_size"],
        learning_starts=config_data["learning_starts"],
        batch_size=config_data["batch_size"],
        tau=config_data["tau"],
        gamma=config_data["gamma"],
        train_freq=config_data["train_freq"],
        gradient_steps=config_data["gradient_steps"],
        seed=config_data["seed"],
        render=True,
        trained_model=None,
        # trained_model=trained_model,
        scene_graph=BallBasketSceneGraph(),
        experiment_save_dir=config_data['human_data_save_path'],
        # credit_assignment=config_data["credit_assignment"]
    )

    print(f"Model Policy = " + str(model.policy))

    if not config_data["load_model"]:
        # model.learn(
        #     config_data["steps"],
        # )
        # mean_reward, std_reward = evaluate_policy(
        #     model, env, n_eval_episodes=20, render=False
        # )
        # print(f"After Training: Mean reward: {mean_reward} +/- {std_reward:.2f}")
        train_model(model, config_data, feedback_gui, human_feedback, env)

    else:
        del model
        model = ActiveTamerRLSACOptimBallBasket.load(config_data["load_model"], env=env)
        print("Loaded pretrained model")
        train_model(model, config_data, feedback_gui, human_feedback, env)
        # do visualization
        # obs = env.reset()
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # get warning to update gpu when using gpu
        
        # for i in range(100):
        #     # model_action, _ = trained_model.actor.action_log_prob(torch.tensor(obs).view(1, -1).to(device))
        #     model_action, _ = model.predict(torch.tensor(obs).view(1, -1).to(device))
        #     model_action = torch.from_numpy(model_action)

        #     obs, reward, done, _ = env.step(model_action[0].cpu().detach().numpy())
        #     env.render()

        #     print("action", model_action)
        #     # print("observation [x, y]", obs)
        #     print("reward", reward)

        #     if done:
        #         print("DONEEEE!!")
        #         print(reward)
        #         obs = env.reset()
        #         print(obs)
        #         # break

if __name__ == "__main__":
    # msg = "Overwrite config params"
    # parser = argparse.ArgumentParser(description = msg)
    # parser.add_argument("--seed", type=int, default=None)

    # args = parser.parse_args()
    # main(args)
    main()

