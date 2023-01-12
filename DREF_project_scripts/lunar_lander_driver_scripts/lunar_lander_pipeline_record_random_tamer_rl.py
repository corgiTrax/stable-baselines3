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

def get_abstract_state(curr_state_vec):
    y_state = -1
    y_obs = float(curr_state_vec[0][1])
    x_state = -1
    x_obs = float(curr_state_vec[0][0])

    if y_obs < 0.5:
        y_state = 2
    elif y_obs > 1:
        y_state = 0
    else:
        y_state = 1

    if x_obs < -0.3:
        x_state = 0
    elif x_obs > 0.3:
        x_state = 2
    else:
        x_state = 1

    return x_state * 3 + y_state

def get_abstract_state(curr_state_vec):
    y_state = -1
    y_obs = float(curr_state_vec[0][1])
    x_state = -1
    x_obs = float(curr_state_vec[0][0])

    if y_obs < 0.5:
        y_state = 2
    elif y_obs > 1:
        y_state = 0
    else:
        y_state = 1

    if x_obs < -0.3:
        x_state = 0
    elif x_obs > 0.3:
        x_state = 2
    else:
        x_state = 1

    return x_state * 3 + y_state

class LunarLanderSceneGraph:
    agent = {'location': {'x': 0, 'y': 0}, 'action': {'down': 0, 'lateral': 0}}
    flag1 = {'location': {'x': -0.28, 'y': 0.235}}
    flag2 = {'location': {'x': 0.28, 'y': 0.235}}
    mountain = {'location': {'x': 0, 'y': 0}}
    state_counts = collections.Counter()
    max_counts = 0
    curr_graph = None
    total_feedback = 200000
    given_feedback = 0

    def isLeft(self, obj_a, obj_b):
        return obj_a['location']['x'] < obj_b['location']['x']
    
    def onTop(self, obj_a, obj_b):
        return obj_a['location']['y'] > obj_b['location']['y']

    def getStateRank(self):
        curr_graph_count = self.state_counts[tuple(self.curr_graph)]
        rank = 0
        for graph in self.state_counts:
            if self.state_counts[graph] < curr_graph_count:
                rank += 1
        return rank
    
    def midway(self, obj_a):
        return obj_a['location']['y'] < 0.5
    
    def oob_left(self, obj_a):
        return obj_a['location']['x'] < -0.4
    
    def oob_right(self, obj_a):
        return obj_a['location']['x'] > 0.4
    
    def oob_top(self, obj_a):
        return obj_a['location']['y'] > 1.0
    
    def main_engine_on(self, obj_a):
        return obj_a['action']['down'] > 0
    
    def left_engine_on(self, obj_a):
        return obj_a['action']['lateral']  < -0.5
    
    def right_engine_on(self, obj_a):
        return obj_a['action']['lateral']  > 0.5

    def is_upright(self, obj_a):
        return obj_a['orientation'] > -0.5 and obj_a['orientation'] < 0.5 # this is in radians. 0.0 is facing +y on coord plane.
    
    def getCurrGraph(self):
        self.curr_graph = [self.isLeft(self.agent, self.flag1), self.isLeft(self.agent, self.flag2), self.isLeft(self.agent, self.mountain),
                self.onTop(self.agent, self.flag1), self.onTop(self.agent, self.flag2), self.onTop(self.agent, self.mountain), 
                self.is_upright(self.agent), self.main_engine_on(self.agent), self.left_engine_on(self.agent), self.right_engine_on(self.agent)]
        # self.curr_graph = [self.isLeft(self.agent, self.mountain), self.onTop(self.agent, self.mountain), self.midway(self.agent), self.oob_left(self.agent), self.oob_right(self.agent), self.oob_top(self.agent)]
        self.state_counts[tuple(self.curr_graph)] += 1
        self.max_counts = max(self.max_counts, self.state_counts[tuple(self.curr_graph)])
        self.curr_prob = 0.1 * (1 - self.state_counts[tuple(self.curr_graph)] / self.max_counts) * max(1, (10 ** (5 / (self.state_counts[tuple(self.curr_graph)] ** 0.3)) - 0.003 * self.state_counts[tuple(self.curr_graph)]))
        return self.curr_graph
        
    def updateGraph(self, newState, action):
        prev_graph = copy.copy(self.curr_graph)
        self.agent['location'] = {'x': newState[0][0], 'y': newState[0][1]}
        self.agent['action'] = {'down': action[0][0], 'lateral': action[0][0]}
        self.agent['orientation'] = newState[0][4]
        self.given_feedback += 1
        return self.getCurrGraph() != prev_graph, self.curr_prob, self.getStateRank() <= (int(self.total_feedback / self.given_feedback))

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


class LunarLanderSceneGraphV2:
    agent = {'location': {'x': 0, 'y': 0}, 'action': {'down': 0, 'lateral': 0}}
    flag1 = {'location': {'x': -0.28, 'y': 0.235}}
    flag2 = {'location': {'x': 0.28, 'y': 0.235}}
    mountain = {'location': {'x': 0, 'y': 0}}
    num_feedback_given = collections.Counter()
    aRPE_average = collections.Counter()
    curr_graph = None
    total_feedback = 250000 #200000 for frequency based scene graph
    given_feedback = 0
    total_timesteps = 0

    def isLeft(self, obj_a, obj_b):
        return obj_a['location']['x'] < obj_b['location']['x']
    
    def onTop(self, obj_a, obj_b):
        return obj_a['location']['y'] > obj_b['location']['y']
    
    def calculate_ucb(self, graph):
        # file1 = open("ucb_proportions.txt", "a")
        # file1.write(f'{self.aRPE_average[tuple(graph)]}   {0.01 * math.sqrt(2 * self.given_feedback / (self.num_feedback_given[tuple(graph)] + 1))}\n')
        # file1.close()+
        return self.aRPE_average[tuple(graph)] + 0.2 * math.sqrt(2 * self.given_feedback / (self.num_feedback_given[tuple(graph)] + 1))       
    
    def getUCBRank(self):
        curr_graph_ucb1 = self.calculate_ucb(self.curr_graph)
        rank = 0
        for graph in self.num_feedback_given:
            if self.calculate_ucb(graph) > curr_graph_ucb1:
                rank += 1
        return rank
        
    def midway(self, obj_a):
        return obj_a['location']['y'] < 0.5
    
    def oob_left(self, obj_a):
        return obj_a['location']['x'] < -0.4
    
    def oob_right(self, obj_a):
        return obj_a['location']['x'] > 0.4
    
    def oob_top(self, obj_a):
        return obj_a['location']['y'] > 1.0
    
    def main_engine_on(self, obj_a):
        return obj_a['action']['down'] > 0
    
    def left_engine_on(self, obj_a):
        return obj_a['action']['lateral']  < -0.5
    
    def right_engine_on(self, obj_a):
        return obj_a['action']['lateral']  > 0.5

    def is_upright(self, obj_a):
        return obj_a['orientation'] > -0.5 and obj_a['orientation'] < 0.5 # this is in radians. 0.0 is facing +y on coord plane.
    
    def updateRPE(self, human_feedback, human_critic_prediction):
        self.num_feedback_given[tuple(self.curr_graph)] += 1
        self.given_feedback += 1
        self.aRPE_average[tuple(self.curr_graph)] *= (self.num_feedback_given[tuple(self.curr_graph)] - 1)/self.num_feedback_given[tuple(self.curr_graph)]
        self.aRPE_average[tuple(self.curr_graph)] += abs(human_feedback - human_critic_prediction)/self.num_feedback_given[tuple(self.curr_graph)]

    def getCurrGraph(self):
        self.curr_graph = [self.isLeft(self.agent, self.flag1), self.isLeft(self.agent, self.flag2), self.isLeft(self.agent, self.mountain),
                self.onTop(self.agent, self.flag1), self.onTop(self.agent, self.flag2), self.onTop(self.agent, self.mountain), 
                self.is_upright(self.agent), self.main_engine_on(self.agent), self.left_engine_on(self.agent), self.right_engine_on(self.agent)]
        
        return self.curr_graph
        
    def updateGraph(self, newState, action):
        prev_graph = copy.copy(self.curr_graph)
        self.agent['location'] = {'x': newState[0][0], 'y': newState[0][1]}
        self.agent['action'] = {'down': action[0][0], 'lateral': action[0][0]}
        self.agent['orientation'] = newState[0][4]
        self.total_timesteps += 1
        return self.getCurrGraph() != prev_graph, self.getUCBRank() <= 60

def main():
    with open("configs/lunar_lander/active_tamer_rl_sac_record.yaml", "r") as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)

    tensorboard_log_dir = config_data["tensorboard_log_dir"]
    env = gym.make("LunarLanderContinuous-v2")

    human_feedback = HumanFeedback()
    app = QApplication(sys.argv)
    feedback_gui = FeedbackInterface()
    np.set_printoptions(threshold=np.inf)

    policy_kwargs = dict(
        net_arch=[400, 300],
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
        abstract_state=get_abstract_state,
        scene_graph=LunarLanderSceneGraphV2(),
        percent_feedback=0.06,
    )

    print(f"Model Policy = " + str(model.policy))

    if not config_data["load_model"]:
        thread.Thread(
            target=train_model,
            args=[model, config_data, feedback_gui, human_feedback, env],
            name="train_model",
            daemon=True,
        ).start()
        sys.exit(app.exec_())
    else:
        del model
        model_num = config_data["load_model"]
        model = TamerRLSACRecord.load(f"models/TamerSAC_{model_num}.pt", env=env)
        print("Loaded pretrained model")


if __name__ == "__main__":
    main()
