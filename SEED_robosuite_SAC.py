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

from stable_baselines3.active_tamer.sac import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

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


def main(args, env):
    with open("configs/robosuite/sac.yaml", "r") as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)
    
    if args.seed:
        config_data['seed'] = args.seed

    tensorboard_log_dir = config_data["tensorboard_log_dir"]

    # robosuite_config = {
    #     "env_name": "",
    #     "robots": "Sawyer",
    #     "controller_configs": load_controller_config(default_controller="OSC_POSITION"),
    # }

    # env = wrappers.GymWrapper(suite.make(
    #     **robosuite_config,
    #     has_renderer=False,
    #     has_offscreen_renderer=False,
    #     render_camera="agentview",
    #     ignore_done=False,
    #     use_camera_obs=False,
    #     control_freq=20,
    #     reward_scale=100,
    #     hard_reset=False,
    #     prehensile=False,
    # ), keys=['eef_xyz_gripper'])

    env = wrappers.GymWrapper(env, keys=["eef_xy"])
    print(env.action_space)
    print(env.action_dim)
    # keys=['robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel', 'robot0_eef_quat', 
    #         'robot0_gripper_qpos', 'robot0_gripper_qvel', 'robot0_proprio-state']
    np.set_printoptions(threshold=np.inf)

    policy_kwargs = dict(
        net_arch=[128, 128],
    )
    os.makedirs(tensorboard_log_dir, exist_ok=True)

    kwargs = dict(seed=0)
    kwargs.update(dict(buffer_size=1))

    model = SAC(
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
        model = SAC.load(f"models/SAC_{model_num}.pt", env=env)
        print("Loaded pretrained model")


if __name__ == "__main__":
    msg = "Overwrite config params"
    parser = argparse.ArgumentParser(description = msg)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    # env = suite.make(
    #     env_name="Reaching2D",
    #     robots="Panda",
    #     controller_configs=load_controller_config(default_controller="OSC_POSE"),
    #     gripper_types="default",
    #     initialization_noise=None,
    #     table_full_size=(0.65, 0.8, 0.15),
    #     table_friction=(100, 100, 100),
    #     use_camera_obs=False,
    #     use_object_obs=True,
    #     reward_scale=1.0,
    #     has_renderer=True,
    #     has_offscreen_renderer=False,
    #     render_camera="frontview",
    #     ignore_done=False,
    #     hard_reset=True,
    #     camera_names="frontview",
    #     target_half_size=(0.05, 0.05, 0.001), # target width, height, thickness
    #     target_position=(0.0, 0.0), # target position (height above the table)
    #     random_init=True,
    #     random_target=False,
    # )

    env = suite.make(
        env_name="POCReaching",
        robots="Panda",
        controller_configs=load_controller_config(default_controller="OSC_POSE"),
        initialization_noise=None,
        table_full_size=(0.65, 0.8, 0.15),
        table_friction=(100, 100, 100),
        use_camera_obs=False,
        use_object_obs=True,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="frontview",
        ignore_done=False,
        camera_names="frontview",
        random_init=False,
    )

    main(args, env)
