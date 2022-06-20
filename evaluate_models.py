from __future__ import print_function

import glob
import itertools
import os
import random
import sys
from typing import Callable

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from lunar_lander_models import LunarLanderExtractor, LunarLanderStatePredictor
from PyQt5.QtWidgets import *

from stable_baselines3.active_tamer.tamer_sac import TamerSAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.sac.sac import SAC


def main():
    env = gym.make("LunarLanderContinuous-v2")
    # models = sorted(
    #     glob.glob("models/TamerSACOptim_100000_using_env*"), key=lambda x: int(x.split("_")[-1].split(".")[0])
    # )
    models = ["trained-models/LunarLanderContinuous-v2.zip"]
    # models = ["trained-models/LunarLanderContinuous-v2.zip"]
    scores = []
    for model_path in models:
        env.reset()
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }
        curr_model = SAC.load(model_path, env, custom_objects=custom_objects)
        # curr_model = TamerSAC.load(model_path)
        mean_reward, std_reward = evaluate_policy(
            curr_model, env, n_eval_episodes=1000, render=True
        )
        print(
            f"Current Model = {model_path} After Training: Mean reward: {mean_reward} +/- {std_reward:.2f}"
        )
        scores.append(mean_reward)
        del curr_model
    # plt.plot(scores)
    # plt.show()
    # model.save(f'models/TamerSAC_15000.pt')
    # model.save(f'models/SAC_10000.pt')


if __name__ == "__main__":
    main()
