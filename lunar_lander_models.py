import os
# t
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class LunarLanderEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LunarLanderEncoder, self).__init__()

        self.lin_1 = nn.Linear(input_dim, 64)
        self.lin_2 = nn.Linear(64, 128)
        self.lin_3 = nn.Linear(128, 256)
        self.lin_4 = nn.Linear(256, 512)
        self.lin_5 = nn.Linear(512, 256)
        self.lin_6 = nn.Linear(256, 128)
        self.lin_7 = nn.Linear(128, output_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:

        out = F.relu(self.lin_1(observations))
        out = F.relu(self.lin_2(out))
        out = F.relu(self.lin_3(out))
        out = F.relu(self.lin_4(out))
        out = F.relu(self.lin_5(out))
        out = F.relu(self.lin_6(out))
        out = F.relu(self.lin_7(out))
        return out


class LunarLanderDecoder(nn.Module):
    def __init__(self, input_dim, action_dim, output_dim):
        super(LunarLanderDecoder, self).__init__()

        self.lin_1 = nn.Linear(input_dim + action_dim, 32)
        self.lin_2 = nn.Linear(32, 128)
        self.lin_3 = nn.Linear(128, 128)
        self.lin_4 = nn.Linear(128, output_dim)

    def forward(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        observations = torch.cat((observations, actions), dim=1)
        out = F.relu(self.lin_1(observations))
        out = F.relu(self.lin_2(out))
        out = F.relu(self.lin_3(out))
        out = F.relu(self.lin_4(out))
        return out


class LunarLanderStatePredictor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(LunarLanderStatePredictor, self).__init__()
        self.encoder = LunarLanderEncoder(state_dim, hidden_dim)
        self.decoder = LunarLanderDecoder(hidden_dim, action_dim, state_dim)

    def forward(
        self, curr_state: torch.Tensor, curr_action: torch.Tensor
    ) -> torch.Tensor:
        hidden_state = self.encoder(curr_state)
        next_state = self.decoder(hidden_state, curr_action)
        return next_state


class LunarLanderExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super(LunarLanderExtractor, self).__init__(observation_space, features_dim=1)

        self.input_features = observation_space.shape[0]
        self.hidden_dim = 32
        self.extractor = LunarLanderEncoder(self.input_features, self.hidden_dim)
        self._features_dim = self.hidden_dim
        model_path = "models/lunar_lander_encoder.pt"
        if os.path.exists(model_path):
            self.extractor.load_state_dict(torch.load(model_path))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.extractor(observations)
