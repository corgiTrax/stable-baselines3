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
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.human_feedback import HumanFeedback
from stable_baselines3.common.online_learning_interface import FeedbackInterface
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class LunarLanderEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LunarLanderEncoder, self).__init__()

        self.lin_1 = nn.Linear(input_dim, 32)
        self.lin_2 = nn.Linear(32, 128)
        self.lin_3 = nn.Linear(128, 128)
        self.lin_4 = nn.Linear(128, output_dim)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:

        out = F.relu(self.lin_1(observations))
        out = F.relu(self.lin_2(out))
        out = F.relu(self.lin_3(out))
        out = F.relu(self.lin_4(out))
        return out

class LunarLanderDecoder(nn.Module):
    def __init__(self, input_dim, action_dim, output_dim):
        super(LunarLanderEncoder, self).__init__()

        self.lin_1 = nn.Linear(input_dim + action_dim, 32)
        self.lin_2 = nn.Linear(32, 128)
        self.lin_3 = nn.Linear(128, 128)
        self.lin_4 = nn.Linear(128, output_dim)
    
    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        observations = torch.cat(observations, actions, dim=1)
        out = F.relu(self.lin_1(observations))
        out = F.relu(self.lin_2(out))
        out = F.relu(self.lin_3(out))
        out = F.relu(self.lin_4(out))
        return out

class LunarLanderStatePredictor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(LunarLanderStatePredictor, self).__init__()
        self.encoder = LunarLanderEncoder(state_dim, hidden_dim)
        self.decoder = LunarLanderDecoder(hidden_dim + action_dim, state_dim)

    def forward(self, curr_state: torch.Tensor, curr_action: torch.Tensor) -> torch.Tensor:
        hidden_state = self.encoder(curr_state)
        next_state = self.decoder(hidden_state, curr_action)
        return next_state


class LunarLanderExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super(LunarLanderExtractor, self).__init__(observation_space, features_dim=1)

        self.input_features = observation_space._shape[0]
        self.hidden_dim = 64
        # self.extractor = nn.Sequential(
        #     nn.Linear(self.input_features, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, self.hidden_dim),
        #     nn.ReLU(),
        # )
        self.extractor = LunarLanderEncoder(self.input_features, self.hidden_dim)
        self._features_dim = self.hidden_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.extractor(observations)


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
        render=False,
    )

    if not config_data['load_model']:
        model.learn(
            config_data["steps"],
            human_feedback_gui=feedback_gui,
            human_feedback=human_feedback,
        )
    else:
        print("LOADED PRETRAINED MODEL MODEL")
        model_num = config_data['load_model']
        model.load(f'models/TamerSAC_{model_num}.pt')

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20, render=True)
    model.save('model/TAMER_SAC_10001.pt')
    print(f"After Training: Mean reward: {mean_reward} +/- {std_reward:.2f}")


if __name__ == "__main__":
    main()
