from __future__ import print_function

import itertools
import os
import random
import sys
from typing import Callable

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from PyQt5.QtWidgets import *

from stable_baselines3.active_tamer.tamer_sac import TamerSAC
from stable_baselines3.sac.sac import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.human_feedback import HumanFeedback
from stable_baselines3.common.online_learning_interface import FeedbackInterface
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from lunar_lander_models import LunarLanderExtractor, LunarLanderStatePredictor

def main():
    with open("configs/tamer_sac.yaml", "r") as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)

    tensorboard_log_dir = config_data["tensorboard_log_dir"]
    env = gym.make("LunarLanderContinuous-v2")

    human_feedback = HumanFeedback()
    # app = QApplication(sys.argv)
    # feedback_gui = FeedbackInterface()
    feedback_gui = None
    np.set_printoptions(threshold=np.inf)

    policy_kwargs = dict(
        features_extractor_class=LunarLanderExtractor,
    )
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    model1000 = TamerSAC.load(f'models/TamerSAC_1000.pt', env)
    model2000 = TamerSAC.load(f'models/TamerSAC_10000.pt', env)
    env.reset()
    mean_reward, std_reward = evaluate_policy(model1000, env, n_eval_episodes=20, render=False)
    print(f"After Training: Mean reward: {mean_reward} +/- {std_reward:.2f}")

    # env.reset()
    mean_reward, std_reward = evaluate_policy(model2000, env, n_eval_episodes=20, render=False)
    print(f"After Training: Mean reward: {mean_reward} +/- {std_reward:.2f}")
    del model2000
    del model1000
    # model.save(f'models/TamerSAC_15000.pt')
    # model.save(f'models/SAC_10000.pt')


if __name__ == "__main__":
    main()
