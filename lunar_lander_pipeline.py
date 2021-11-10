from __future__ import print_function

import gym
import numpy as np
import itertools
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Callable
import random
import yaml
from PyQt5.QtWidgets import *

from stable_baselines3.active_tamer.tamer_sac import TamerSAC
from stable_baselines3.common.human_feedback import HumanFeedback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.online_learning_interface import FeedbackInterface
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class LunarLanderExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super(LunarLanderExtractor, self).__init__(observation_space, features_dim=1)
        
        self.input_features = observation_space._shape[0]
        self.hidden_dim = 64
        self.extractor = nn.Sequential(
                        nn.Linear(self.input_features, 32),
                        nn.ReLU(),
                        nn.Linear(32, 128),
                        nn.ReLU(),
                        nn.Linear(128, 128),
                        nn.ReLU(),
                        nn.Linear(128, self.hidden_dim),
                        nn.ReLU(),
                    )
        self._features_dim = self.hidden_dim
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.extractor(observations)

def main():
    with open("configs/tamer_sac.yaml", "r") as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)

    tensorboard_log_dir = config_data["tensorboard_log_dir"]
    env = gym.make('LunarLanderContinuous-v2')
    
    human_feedback = HumanFeedback()
    # app = QApplication(sys.argv)
    # feedback_gui = FeedbackInterface()
    feedback_gui = None
    np.set_printoptions(threshold=np.inf)

    policy_kwargs = dict(
        features_extractor_class=LunarLanderExtractor,
    )
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    model = TamerSAC(
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
    )

    model.learn(config_data["steps"], human_feedback_gui=feedback_gui, human_feedback=human_feedback)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
    print(f"After Training: Mean reward: {mean_reward} +/- {std_reward:.2f}")

if __name__ == "__main__":
    main()