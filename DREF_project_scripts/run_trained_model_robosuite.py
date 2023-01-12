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

if __name__ == "__main__":

    # # Create dict to hold options that will be passed to env creation call
    # options = {}

    # # print welcome info
    # print("Welcome to robosuite v{}!".format(suite.__version__))
    # print(suite.__logo__)

    # # Choose environment and add it to options
    # options["env_name"] = choose_environment()

    # # If a multi-arm environment has been chosen, choose configuration and appropriate robot(s)
    # if "TwoArm" in options["env_name"]:
    #     # Choose env config and add it to options
    #     options["env_configuration"] = choose_multi_arm_config()

    #     # If chosen configuration was bimanual, the corresponding robot must be Baxter. Else, have user choose robots
    #     if options["env_configuration"] == "bimanual":
    #         options["robots"] = "Baxter"
    #     else:
    #         options["robots"] = []

    #         # Have user choose two robots
    #         print("A multiple single-arm configuration was chosen.\n")

    #         for i in range(2):
    #             print("Please choose Robot {}...\n".format(i))
    #             options["robots"].append(choose_robots(exclude_bimanual=True))

    # # Else, we simply choose a single (single-armed) robot to instantiate in the environment
    # else:
    #     options["robots"] = choose_robots(exclude_bimanual=True)

    # # Choose controller
    # controller_name = choose_controller()

    # # Load the desired controller
    # options["controller_configs"] = load_controller_config(default_controller=controller_name)

    # # Help message to user
    # print()
    # print('Press "H" to show the viewer control panel.')
    
    robosuite_config = {
        "env_name": "Reaching",
        "robots": "Sawyer",
        "controller_configs": load_controller_config(default_controller="OSC_POSITION"),
    }

    env = wrappers.GymWrapper(suite.make(
        **robosuite_config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="agentview",
        ignore_done=False,
        use_camera_obs=False,
        reward_shaping=False,
        control_freq=20,
        reward_scale=10,
        hard_reset=False,
        render_gpu_device_id=0,
    ), keys=['robot0_eef_pos_xy'])

    # env = Monitor(env)
    # env = DummyVecEnv([lambda: env])

    # agentview_image
    
    print(env.action_space)
    print(env.observation_space)
    obs = env.reset()
    env.render()
    env.viewer.set_camera(camera_id=1)

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

    trained_model = SAC.load(
        "trained-models/RobosuiteReaching.pt", env, custom_objects=custom_objects, **kwargs
    )
    # do visualization
    for i in range(10000):
        # action = np.random.uniform(low, high)
        model_action, _ = trained_model.actor.action_log_prob(torch.tensor(obs).view(1, -1).to(device))
        print(model_action[0])
        # print(model_action[0].cpu().detach().numpy())
        obs, reward, done, _ = env.step(model_action[0].cpu().detach().numpy())
        print(obs)
        # env.render()
        env.render()
        if done:
            print("DONEEEE!!")
            print(reward)
            obs = env.reset()
            print(obs)
