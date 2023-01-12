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

from stable_baselines3.sac.sac_record_robot_ballbasket import SACRecord # CHANGED
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


import robosuite as suite
from robosuite import wrappers
from robosuite import load_controller_config


sys.path.insert(1, '/home/robot/perls2')
# perls2 modules
from demos.sawyer_osc_3d import OpSpaceLineXYZ
from real_sawyer_env import RealSawyerBallBasketEnv


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
    with open("configs/real_robot/sac_record.yaml", "r") as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)
    
    # if args.seed:
    #     config_data['seed'] = args.seed

    tensorboard_log_dir = config_data["tensorboard_log_dir"]

    ### initialize environment ###
    parser = argparse.ArgumentParser(
        description="Test controllers and measure errors.")
    parser.add_argument('--world', default=None, help='World type for the demo, uses config file if not specified', choices=['Bullet', 'Real'])
    parser.add_argument('--robot', default='sawyer', help='Robot type overrides config', choices=['panda', 'sawyer'])
    parser.add_argument('--ctrl_type',
                        default="EEImpedance",
                        help='Type of controller to test')
    parser.add_argument('--demo_type',
                        default="Line",
                        help='Type of menu to run.')
    parser.add_argument('--test_fn',
                        default='set_ee_pose',
                        help='Function to test',
                        choices=['set_ee_pose', 'move_ee_delta', 'set_joint_delta', 'set_joint_positions', 'set_joint_torques', 'set_joint_velocities'])
    parser.add_argument('--path_length', type=float,
                        default=None, help='length in m of path')
    parser.add_argument('--delta_val',
                        default=[0.001, 0.001, 0.001], type=float,
                        help="Max step size (m or rad) to take for demo.")
    parser.add_argument('--axis',
                        default='x', type=str,
                        choices=['x', 'y', 'z'],
                        help='axis for demo. Position direction for Line or rotation axis for Rotation')
    parser.add_argument('--num_steps', default=1, type=int,
                        help="max steps for demo.")
    parser.add_argument('--plot_pos', action="store_true",
                        help="whether to plot positions of demo.")
    parser.add_argument('--plot_error', action="store_true",
                        help="whether to plot errors.")
    parser.add_argument('--save', action="store_true",
                        help="whether to store data to file")
    parser.add_argument('--demo_name', default=None,
                        type=str, help="Valid filename for demo.")
    parser.add_argument('--save_fig', action="store_true",
                        help="whether to save pngs of plots")
    parser.add_argument('--fix_ori', action="store_true", default=True,
                        help="fix orientation for move_ee_delta")
    parser.add_argument('--fix_pos', action="store_true",
                        help="fix position for move_ee_delta")
    parser.add_argument('--config_file', default='/home/robot/perls2/demos/demo_control_cfg.yaml', help='absolute filepath for config file.')
    parser.add_argument('--cycles', type=int, default=1, help="num times to cycle path (only for square)")
    args = parser.parse_args()
    kwargs = vars(args)

    driver = OpSpaceLineXYZ(**kwargs)

    # env = RealSawyerBallBasketEnv(driver, random_init=False)
    env = RealSawyerBallBasketEnv(driver, random_init=True)

    np.set_printoptions(threshold=np.inf)

    policy_kwargs = dict(
        net_arch=[128, 128],
    )
    os.makedirs(tensorboard_log_dir, exist_ok=True)

    kwargs = dict(seed=0)
    kwargs.update(dict(buffer_size=1))

    while os.path.exists(config_data['data_save_path']):
        config_data['data_save_path'] = "/".join(config_data['data_save_path'].split("/")[:-1]) + '/run_' + str(int(random.random() * 1000000000))

    model = SACRecord(
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
        experiment_save_dir=config_data["data_save_path"],
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
    # msg = "Overwrite config params"
    # parser = argparse.ArgumentParser(description = msg)
    # parser.add_argument("--seed", type=int, default=None)

    # args = parser.parse_args()
    # main(args)
    main()