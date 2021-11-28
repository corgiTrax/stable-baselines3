import copy
import os
import random
import sys
from logging import Handler
from typing import final

import gym
import igibson
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from igibson.envs.behavior_env import BehaviorEnv
from PyQt5.QtWidgets import *

from stable_baselines3.common.human_feedback import HumanFeedback
from stable_baselines3.common.online_learning_interface import FeedbackInterface
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

app = None
feedback_gui = None


class OLNet_taskObs(nn.Module):
    def __init__(self, task_obs_dim=456, proprioception_dim=20, num_actions=11):
        super(OLNet_taskObs, self).__init__()
        # image feature
        self.fc1 = nn.Linear(task_obs_dim + proprioception_dim, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, num_actions)

    def forward(self, task_obs, proprioceptions):
        x = torch.cat((task_obs, proprioceptions), dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, device, num_actions=11):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)

        extractors = {}
        self.device = device
        total_concat_size = 0
        feature_size = 128
        for key, subspace in observation_space.spaces.items():
            if key in ["proprioception", "task_obs"]:
                extractors[key] = nn.Sequential(
                    nn.Linear(subspace.shape[0], feature_size), nn.ReLU()
                )
            elif key in ["rgb", "highlight", "depth", "seg", "ins_seg"]:
                n_input_channels = subspace.shape[2]  # channel last
                cnn = nn.Sequential(
                    nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                test_tensor = torch.zeros(
                    [subspace.shape[2], subspace.shape[0], subspace.shape[1]]
                )
                with torch.no_grad():
                    n_flatten = cnn(test_tensor[None]).shape[1]
                fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
                extractors[key] = nn.Sequential(cnn, fc)
            elif key in ["scan"]:
                n_input_channels = subspace.shape[1]  # channel last
                cnn = nn.Sequential(
                    nn.Conv1d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
                    nn.ReLU(),
                    nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                test_tensor = torch.zeros([subspace.shape[1], subspace.shape[0]])
                with torch.no_grad():
                    n_flatten = cnn(test_tensor[None]).shape[1]
                fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
                extractors[key] = nn.Sequential(cnn, fc)
            else:
                raise ValueError("Unknown observation key: %s" % key)
            total_concat_size += feature_size

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

        self.fc1 = nn.Linear(total_concat_size, total_concat_size * 2)
        self.fc2 = nn.Linear(total_concat_size * 2, total_concat_size * 4)
        self.fc3 = nn.Linear(total_concat_size * 4, total_concat_size * 8)
        self.fc4 = nn.Linear(total_concat_size * 8, total_concat_size * 4)
        self.fc5 = nn.Linear(total_concat_size * 4, total_concat_size * 2)
        self.fc6 = nn.Linear(total_concat_size * 2, num_actions)

        self.fc = nn.Sequential(
            nn.Linear(total_concat_size, total_concat_size * 2),
            nn.ReLU(),
            nn.Linear(total_concat_size * 2, total_concat_size * 4),
            nn.ReLU(),
            nn.Linear(total_concat_size * 4, total_concat_size * 8),
            nn.ReLU(),
            nn.Linear(total_concat_size * 8, total_concat_size * 4),
            nn.ReLU(),
            nn.Linear(total_concat_size * 4, total_concat_size * 2),
            nn.ReLU(),
            nn.Linear(total_concat_size * 2, num_actions),
        )

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        curr_embedded_state = torch.cat(encoded_tensor_list, dim=1)
        return self.fc(curr_embedded_state)


def convert_obs_to_torch(obs, device):
    for key in obs:
        obs[key] = torch.tensor(obs[key].copy()).unsqueeze(0).float().to(device)
        if key in ["rgb", "highlight", "depth", "seg", "ins_seg"]:
            obs[key] = obs[key].permute((0, 3, 1, 2))
        elif key in ["scan"]:
            obs[key] = obs[key].permute((0, 2, 1))
    return obs


def concatenate_obs(obs, device):
    final_obs = copy.copy(obs[0])
    for ob in obs[1:]:
        for key in ob:
            final_obs[key] = torch.cat((final_obs[key], ob[key]), dim=0)

    return final_obs


def concatenate_obs_actions(sampled_obs_actions, device):
    final_actions = sampled_obs_actions[0][1]
    final_actions = np.expand_dims(final_actions, axis=0)
    for _, action in sampled_obs_actions[1:]:
        final_actions = np.concatenate(
            (final_actions, np.expand_dims(action, axis=0)), axis=0
        )
    final_actions = torch.tensor(final_actions.copy()).float().to(device)
    final_obs = concatenate_obs([i[0] for i in sampled_obs_actions], device)
    return final_obs, final_actions


def train_ol_model(
    ol_agent,
    env,
    device,
    learning_rate,
    robot,
    iterations,
    train_every=1,
    max_replay_buffer=3072,
    batch_size=16,
):
    optimizer = None
    app = QApplication(sys.argv)
    feedback_gui = FeedbackInterface()
    human_feedback = HumanFeedback(robot, feedback_gui)
    replay_buffer = []
    obs = env.reset()
    for iteration in range(iterations):
        obs = env.reset()
        total_reward = 0
        done = False
        paused = False
        while not done:
            task_obs = (
                torch.tensor(obs["task_obs"], dtype=torch.float32)
                .unsqueeze(0)
                .to(device)
            )
            proprioception = (
                torch.tensor(obs["proprioception"], dtype=torch.float32)
                .unsqueeze(0)
                .to(device)
            )
            if not ol_agent:
                ol_agent = CustomCombinedExtractor(
                    env.observation_space, device, 11
                ).to(device)
                ol_agent.train()
                optimizer = optim.Adam(ol_agent.parameters())

            optimizer.zero_grad()
            obs = convert_obs_to_torch(obs, device)
            action = ol_agent(obs)
            a = action.cpu().detach().numpy().squeeze(0)
            curr_keyboard_feedback = human_feedback.return_human_keyboard_feedback()
            if curr_keyboard_feedback:
                if "Pause" in str(curr_keyboard_feedback):
                    paused = not paused
                elif "Reset" in str(curr_keyboard_feedback):
                    obs = env.reset()
                    total_reward = 0
                    done = False
                    paused = False
                elif type(curr_keyboard_feedback) == list:
                    error = np.array(curr_keyboard_feedback) * learning_rate
                    label_action = a + error
                    if len(replay_buffer) == max_replay_buffer:
                        replay_buffer = replay_buffer[1:]
                    replay_buffer.append([obs, label_action])
                    print("Received Feedback: " + str(curr_keyboard_feedback))

            if not paused:
                obs, reward, done, info = env.step(a)
            else:
                reward = 0

            if iteration % train_every == 0 and len(replay_buffer) > 0:
                training_samples = random.sample(
                    replay_buffer, min(len(replay_buffer), batch_size)
                )
                obs_batch, label_actions = concatenate_obs_actions(
                    training_samples, device
                )
                actions = ol_agent(obs_batch)
                loss = nn.MSELoss()(actions, label_actions)
                feedback_gui.updateLoss(loss.item())
                loss.backward()
                optimizer.step()

            total_reward += float(reward)
            feedback_gui.updateReward(float(total_reward))


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    iterations = 10000
    ol_agent = None
    config_file = "behavior_full_observability_fetch.yaml"
    env = BehaviorEnv(
        config_file=os.path.join("configs/", config_file),
        mode="gui_interactive",
        action_timestep=1 / 30.0,
        physics_timestep=1 / 300.0,
        action_filter="all",
    )

    train_ol_model(
        ol_agent,
        env,
        device,
        0.1,
        yaml.load(open(f"configs/{config_file}", "r"), Loader=yaml.FullLoader)["robot"],
        iterations,
    )
