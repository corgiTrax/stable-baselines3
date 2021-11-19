from __future__ import print_function

import itertools
import os
import random
import sys
from typing import Callable

import gym
import numpy as np
import torch
import threading as thread
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

def train_model(model, config_data, feedback_gui, human_feedback):
    model.learn(
        config_data["steps"], 
        human_feedback_gui=feedback_gui,
        human_feedback=human_feedback,
    )
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20, render=False)
    print(f"After Training: Mean reward: {mean_reward} +/- {std_reward:.2f}")

def main():
    with open("configs/tamer_sac.yaml", "r") as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)

    tensorboard_log_dir = config_data["tensorboard_log_dir"]
    env = gym.make("LunarLanderContinuous-v2")

    human_feedback = HumanFeedback()
    app = QApplication(sys.argv)
    feedback_gui = FeedbackInterface()
    # feedback_gui = None
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

    # model = SAC(
    #     config_data["policy_name"],
    #     env,
    #     verbose=config_data["verbose"],
    #     tensorboard_log=tensorboard_log_dir,
    #     policy_kwargs=policy_kwargs,
    #     learning_rate=config_data["learning_rate"],
    #     buffer_size=config_data["buffer_size"],
    #     learning_starts=config_data["learning_starts"],
    #     batch_size=config_data["batch_size"],
    #     tau=config_data["tau"],
    #     gamma=config_data["gamma"],
    #     train_freq=config_data["train_freq"],
    #     gradient_steps=config_data["gradient_steps"],
    #     seed=config_data["seed"],
    # )

    if not config_data['load_model']:
        thread.Thread(
            target=train_model,
            args=[model, config_data, feedback_gui, human_feedback],
            name="render_environment",
            daemon=True,
        ).start()
        sys.exit(app.exec_())
    else:
        del model
        model_num = config_data['load_model']
        model = TamerSAC.load(f'models/TamerSAC_{model_num}.pt', env=env)
        print("LOADED PRETRAINED MODEL MODEL")

    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20, render=False)
    # print(f"After Training: Mean reward: {mean_reward} +/- {std_reward:.2f}")


if __name__ == "__main__":
    main()
