from __future__ import print_function

import itertools
import os
import random
import sys
import threading as thread
from typing import Callable

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from lunar_lander_models import LunarLanderExtractor, LunarLanderStatePredictor
from PyQt5.QtWidgets import *

from stable_baselines3.active_tamer.active_tamerRL_sac_optim import (
    ActiveTamerRLSACOptim,
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.sac.sac import SAC


class LunarLanderSceneGraph:
    agent = {'location': {'x': 0, 'y': 0}}
    flag1 = {'location': {'x': -0.28, 'y': 0.235}}
    flag2 = {'location': {'x': 0.28, 'y': 0.235}}
    mountain = {'location': {'x': 0, 'y': 0}}

    def isLeft(self, obj_a, obj_b):
        return obj_a['location']['x'] < obj_b['location']['x']
    
    def onTop(self, obj_a, obj_b):
        return obj_a['location']['y'] > obj_b['location']['y']
    
    def getCurrGraph(self):
        return [self.isLeft(self.agent, self.flag1), self.isLeft(self.agent, self.flag2), self.isLeft(self.agent, self.mountain),
                self.onTop(self.agent, self.flag1), self.onTop(self.agent, self.flag2), self.onTop(self.agent, self.mountain)]
    
    def updateGraph(self, newState):
        prev_graph = self.getCurrGraph()
        self.agent['location'] = {'x': newState[0], 'y': newState[1]}
        return self.getCurrGraph() == prev_graph



curr_env_graph = LunarLanderSceneGraph()
env = gym.make("LunarLanderContinuous-v2")

for i in range(100000):
    observation = env.reset()
    env.render()
    print(curr_env_graph.updateGraph(observation))
