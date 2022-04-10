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
from lunar_lander_models import LunarLanderStatePredictor
from PyQt5.QtWidgets import *


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = gym.make("LunarLanderContinuous-v2")
    obs = env.reset()

    state_predictor = LunarLanderStatePredictor(
        state_dim=len(obs), action_dim=2, hidden_dim=16
    ).to(device)
    observed_states = None
    actions = None

    epochs = 100
    batch_size = 8
    optimizer = optim.SGD(state_predictor.parameters(), lr=0.001, momentum=0.9)
    loss = nn.MSELoss()
    curr_cumulative_loss = 0

    for epoch in range(epochs * batch_size):

        curr_action_random = np.array(env.action_space.sample())
        if observed_states is not None:
            observed_states = np.append(observed_states, obs[np.newaxis], axis=0)
            actions = np.append(actions, curr_action_random[np.newaxis], axis=0)
        else:
            observed_states = obs[np.newaxis]
            actions = curr_action_random[np.newaxis]

        if len(observed_states) == (batch_size + 1):
            optimizer.zero_grad()

            tensor_state = torch.tensor(observed_states, dtype=torch.float32).to(device)
            tensor_state_x = tensor_state[:batch_size]
            tensor_state_y = tensor_state[1:]
            tensor_action = torch.tensor(actions[:batch_size], dtype=torch.float32).to(
                device
            )

            predictions = state_predictor(tensor_state_x, tensor_action)
            curr_loss = loss(predictions, tensor_state_y)
            curr_loss.backward()
            optimizer.step()
            curr_cumulative_loss += curr_loss

            observed_states = None
            actions = None

            if epoch % 1000 == 0:
                print(curr_cumulative_loss / 1000)
                curr_cumulative_loss = 0

        obs, _, done, _ = env.step(curr_action_random)
        if done:
            obs = env.reset()

    torch.save(state_predictor.encoder.state_dict(), "models/lunar_lander_encoder.pt")


if __name__ == "__main__":
    main()
