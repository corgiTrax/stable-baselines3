from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from robosuite import wrappers
import torch
from stable_baselines3.active_tamer.sac import SAC
import sys
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    # VecEnv,
    # VecNormalize,
    # VecTransposeImage,
    # is_vecenv_wrapped,
    # unwrap_vec_normalize,
)
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":

    robosuite_config = {
        "env_name": "BallBasket",
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
        control_freq=20,
        reward_scale=100,
        hard_reset=False,
        prehensile=False,
    ), keys=['eef_xyz_gripper'])
    
    print(env.action_space)
    print(env.observation_space)
    obs = env.reset()
    # env.render()
    # env.viewer.set_camera(camera_id=0)

    # low, high = env.action_spec

    kwargs = dict(seed=0)
    kwargs.update(dict(buffer_size=1))
    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8
    custom_objects = {}
    if newer_python_version:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # trained_model = SAC.load(
    #     "trained-models/BallReaching.pt", env, custom_objects=custom_objects, **kwargs
    # )
    # do visualization
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(100):
        # action = np.random.uniform(low, high)
        # model_action, _ = trained_model.actor.action_log_prob(torch.tensor(obs).view(1, -1).to(device))
        # print(model_action[0])
        # print(model_action[0].cpu().detach().numpy())
        # obs, reward, done, _ = env.step(model_action[0].cpu().detach().numpy())
        # print(obs)
        # env.render()
        # env.render()
        # if done:
            # print("DONEEEE!!")
            # print(reward)
            # obs = env.reset()
            # print(obs)
        print(i)
        ax.scatter(obs[0],obs[1],obs[2])
        obs = env.reset()
    plt.show()
