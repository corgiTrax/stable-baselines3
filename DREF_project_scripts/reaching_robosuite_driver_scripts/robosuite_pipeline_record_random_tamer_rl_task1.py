from __future__ import print_function

import itertools
import os
import random
import sys
import threading as thread
from typing import Callable
from gevent import config
import copy
import math
import gym
import numpy as np
import torch
import collections
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from lunar_lander_models import LunarLanderExtractor, LunarLanderStatePredictor
from PyQt5.QtWidgets import *

from stable_baselines3.active_tamer.tamerRL_sac_record import (
    TamerRLSACRecord,
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.sac.sac import SAC
from stable_baselines3.common.human_feedback import HumanFeedback
from stable_baselines3.common.online_learning_interface import FeedbackInterface

import robosuite as suite
from robosuite import wrappers
from robosuite import load_controller_config

class ReachingSceneGraph:
    agent = {'location': {'x': 0, 'y': 0}}
    num_feedback_given = collections.Counter()
    aRPE_average = collections.Counter()
    curr_graph = None
    total_feedback = 250000 #200000 for frequency based scene graph
    given_feedback = 0
    total_timesteps = 0
    
    def calculate_ucb(self, graph):
        return self.aRPE_average[tuple(graph)] + 0.2 * math.sqrt(2 * self.given_feedback / (self.num_feedback_given[tuple(graph)] + 1))       
    
    def getUCBRank(self):
        curr_graph_ucb1 = self.calculate_ucb(self.curr_graph)
        rank = 0
        for graph in self.num_feedback_given:
            if self.calculate_ucb(graph) > curr_graph_ucb1:
                rank += 1
        return rank

    def right(self, obj_a):
        return obj_a['location']['x'] < -0.05

    def center(self, obj_a):
        return obj_a['location']['x'] > -0.05 and obj_a['location']['x'] < 0.05

    def left(self, obj_a):
        return obj_a['location']['x'] > 0.05

    
    def top(self, obj_a):
        return obj_a['location']['y'] < -0.125

    def middle(self, obj_a):
        return obj_a['location']['y'] > -0.125 and obj_a['location']['y'] < 0.125

    def bottom(self, obj_a):
        return obj_a['location']['y'] > 0.125
    

    def updateRPE(self, human_feedback, human_critic_prediction):
        self.num_feedback_given[tuple(self.curr_graph)] += 1
        self.given_feedback += 1
        self.aRPE_average[tuple(self.curr_graph)] *= (self.num_feedback_given[tuple(self.curr_graph)] - 1)/self.num_feedback_given[tuple(self.curr_graph)]
        self.aRPE_average[tuple(self.curr_graph)] += abs(human_feedback - human_critic_prediction)/self.num_feedback_given[tuple(self.curr_graph)]

    def getCurrGraph(self):
        self.curr_graph = [self.right(self.agent), self.center(self.agent), self.left(self.agent), self.top(self.agent), 
                            self.middle(self.agent), self.bottom(self.agent)]
        
        return self.curr_graph
        
    def updateGraph(self, newState, action):
        prev_graph = copy.deepcopy(self.curr_graph)
        self.agent['location'] = {'x': newState[0][0], 'y': newState[0][1]}
        self.total_timesteps += 1
        return self.getCurrGraph() != prev_graph, self.getUCBRank() <= 4

def train_model(model, config_data, feedback_gui, human_feedback, env):
    model.learn(
        config_data["steps"],
        human_feedback_gui=feedback_gui,
        human_feedback=human_feedback,
    )
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=20, render=True
    )
    print(f"After Training: Mean reward: {mean_reward} +/- {std_reward:.2f}")

def main():
    with open("configs/robosuite/tamer_sac_record.yaml", "r") as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)

    tensorboard_log_dir = config_data["tensorboard_log_dir"]

    robosuite_config = {
        "env_name": "Reaching",
        "robots": "Sawyer",
        "controller_configs": load_controller_config(default_controller="OSC_POSITION"),
    }

    env = wrappers.GymWrapper(suite.make(
        **robosuite_config,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="agentview",
        camera_names="birdview",
        camera_heights=1024,
        camera_widths=1024,
        ignore_done=False,
        use_camera_obs=True,
        reward_shaping=False,
        control_freq=20,
        reward_scale=100,
        hard_reset=False,
        render_gpu_device_id=0,
    ), keys=['robot0_eef_pos_xy'])

    # env.viewer.set_camera(camera_id=1)
    env.reset()
    # env.render()
    import time
    time.sleep(1)

    human_feedback = HumanFeedback()
    app = QApplication(sys.argv)
    feedback_gui = FeedbackInterface()
    np.set_printoptions(threshold=np.inf)

    policy_kwargs = dict(
        net_arch=[64, 64],
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
    trained_model = SAC.load(
        config_data["trained_model"], env, custom_objects=custom_objects, **kwargs
    )

    while os.path.exists(config_data['human_data_save_path']):
        config_data['human_data_save_path'] = "/".join(config_data['human_data_save_path'].split("/")[:-1]) + '/participant_' + str(int(random.random() * 1000000000))

    model = TamerRLSACRecord(
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
        experiment_save_dir=config_data['human_data_save_path'],
        render=True,
        trained_model=trained_model,
        scene_graph=ReachingSceneGraph(),
        credit_assignment=config_data['credit_assignment']
    )

    print(f"Model Policy = " + str(model.policy))

    if not config_data["load_model"]:
        # thread.Thread(
        #     target=train_model,
        #     args=[model, config_data, feedback_gui, human_feedback, env],
        #     name="train_model",
        #     daemon=True,
        # ).start()
        # thread.Thread(
        #     target=viz_robosuite,
        #     name="visualize_robosuite",
        #     daemon=True,
        # ).start()
        # sys.exit(app.exec_())
        train_model(model, config_data, feedback_gui, human_feedback, env)
    else:
        del model
        model = TamerRLSACRecord.load(f'models/{config_data["load_model"]}', env=env)
        print("Loaded pretrained model")
        print(model)


if __name__ == "__main__":
    main()
