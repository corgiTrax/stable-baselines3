from __future__ import print_function

import itertools
import os
import random
import sys
import threading as thread
import argparse
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

from stable_baselines3.active_tamer.tamerRL_sac_optim import TamerRLSACOptim
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.sac.sac import SAC

import robosuite as suite
from robosuite import wrappers
from robosuite import load_controller_config


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


def main(args):
    
    with open("configs/robosuite/tamer_rl_sac.yaml", "r") as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)

    if args.seed:
        config_data['seed'] = args.seed

    tensorboard_log_dir = config_data["tensorboard_log_dir"]

    robosuite_config = {
        "env_name": "Reaching",
        "robots": "Sawyer",
        "controller_configs": load_controller_config(default_controller="OSC_POSITION"),
    }

    env = wrappers.GymWrapper(suite.make(
        **robosuite_config,
        has_renderer=False,
        has_offscreen_renderer=False,
        render_camera="agentview",
        ignore_done=False,
        use_camera_obs=False,
        reward_shaping=False,
        control_freq=20,
        reward_scale=10,
        hard_reset=False,
    ), keys=['robot0_eef_pos_xy'])

    print(env)

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

    model = TamerRLSACOptim(
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
        trained_model=trained_model,
        percent_feedback=config_data['percent_feedback'],
    )

    print(f"Model Policy = " + str(model.policy))

    if not config_data["load_model"]:
        model.learn(
            config_data["steps"],
        )
        mean_reward, std_reward = evaluate_policy(
            model, env, n_eval_episodes=20, render=False
        )
        print(f"After Training: Mean reward: {mean_reward} +/- {std_reward:.2f}")
    else:
        del model
        model_num = config_data["load_model"]
        model = TamerRLSACOptim.load(f"models/TamerSAC_{model_num}.pt", env=env)
        print("Loaded pretrained model")


if __name__ == "__main__":
    msg = "Overwrite config params"
    parser = argparse.ArgumentParser(description = msg)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()
    main(args)
