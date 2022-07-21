import gym
from gym import spaces
import numpy as np
import argparse
import sys
import torch

import pickle5 as pickle


# perls2 modules
sys.path.insert(1, '/home/robot/perls2') # path to perls2 library

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.active_tamer.sac import SAC

from real_sawyer_env import RealSawyerReachingEnv, RealSawyerBallBasketEnv

def run_pretrained(env):
    ### Run pre-trained model ####
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # get warning to update gpu when using gpu

    trained_model = SAC.load(
        "RobosuiteReaching.pt", env, custom_objects=custom_objects, **kwargs
    )
    
    # do visualization
    obs = env.reset()
    
    for i in range(100):
        # model_action, _ = trained_model.actor.action_log_prob(torch.tensor(obs).view(1, -1).to(device))
        model_action, _ = trained_model.predict(torch.tensor(obs).view(1, -1).to(device))
        model_action = torch.from_numpy(model_action)

        obs, reward, done, _ = env.step(model_action[0].cpu().detach().numpy())
        env.render()

        print("action", model_action)
        # print("observation [x, y]", obs)
        print("reward", reward)

        if done:
            print("DONEEEE!!")
            print(reward)
            obs = env.reset()
            print(obs)
            # break

def test_sound(env):

    from playsound import playsound
    import redis

    r = redis.Redis()
    FEEDBACK_REQUEST_KEY = "human_feedback"
    r.set(FEEDBACK_REQUEST_KEY, "0")

    obs = env.reset()

    for i in range(150):
        # action = np.random.uniform(-0.1, 0.1, 4)
        
        if r.get(FEEDBACK_REQUEST_KEY) == b'1':
            playsound("beep.wav", block=False)
            r.set(FEEDBACK_REQUEST_KEY, "0")

        action = np.array([0.05, 0, 0, 0])
        obs, reward, done, _ = env.step(action)
        # env.render()

def record_state(env, axis=0):
    ### Record State ###
    states = []
    actions = np.loadtxt(f"actions{axis}.txt", delimiter=',')
    robosuite_states = np.loadtxt(f"robosuite_states{axis}.txt", delimiter=',')
    for i in range(actions.shape[0]):
        action = actions[i]
        obs, _, _, _ = env.step(action)
        states.append(obs)

    states = np.array(states)

    np.savetxt(f"sawyer_states{axis}.txt", states, delimiter=',', newline='\n', fmt='%1.5f')
    print("-------Finished Actions-------")
    error = np.mean(states - robosuite_states, axis=0)
    states_shifted = states - error
    print(states_shifted - robosuite_states)
    print("correction term", error)

def calibrate_boundary_helper(env):
    while True:
        print("raw", env.driver.get_eef_xyz())
        # print("calib", env.driver.get_eef_xy() - env.origin)
        if env.reward() > 0:
            print("IN TARGET")

def init_env(task=1, random_init=True, pickup_ball=False):
    if task == 1:

        from demos.sawyer_osc_2d import OpSpaceLineXYZ 

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
                            default=[0.001, 0.001], type=float,
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

        env = RealSawyerReachingEnv(driver, random_init=random_init)

        return env

    elif task == 2:

        from demos.sawyer_osc_3d import OpSpaceLineXYZ 

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

        env = RealSawyerBallBasketEnv(driver, random_init=random_init, pickup_ball=pickup_ball)

        return env

def redis_ctrl(env):
    import redis
    r = redis.Redis()

    # read keys
    ACTION_KEY = "action"
    
    while True:
        # Reset environment
        r.set("action", "0, 0, 0, 0") # reset action to neutral
        prev_action = np.zeros(4)

        while True:
            # print("raw", env.driver.get_eef_xyz())
            print("world", env.get_state())
            action = r.get(ACTION_KEY)

            if action != prev_action:
                new_action = str2ndarray(action, (env.action_dim, ))
                obs, reward, done, info = env.step(new_action)
                prev_action = action

            if reward > 0:
                print("target reached!!")

def str2ndarray(array_str, shape):
    """
    Helper function to convert action read from redis to np array
    takes input array_str of form "[[a, b, c],...]"
    returns nparray of specified shape
    """
    
    # array_str = array_str.translate(str.maketrans('','','[]')) # remove brackets
    output = np.fromstring(array_str, dtype=float, sep=',')
    output = output.reshape(shape)
    return output            

if __name__ == "__main__":

    np.random.seed(42)
    env = init_env(task=2, random_init=True, pickup_ball=False)
    # for i in range(5):
    #     env.reset()

    # env = init_env(task=1, random_init=False)
    # eef_xpos = env.get_state()
    # error = eef_xpos - env.origin
    # while abs(env.get_state()[0]) > 0.005:
    #     env.step(np.array([0.05, 0]))
        

    # env.step(np.array([0, 0, 0.2, 0]))
    # env.step(np.array([0, 0, 0.2, 0]))

    # for i in range(10):
    #     env.step(np.array([0.05, 0, 0, 0]))

    # calibrate_boundary_helper(env)

    redis_ctrl(env)

    # for i in range(7):
    #     action = np.array([0, 0, 0.07, 0])
    #     env.step(action)
    #     print(env.get_state())

    # print("[1]", env.robot_interface.close_gripper())
    #print("[2]", env.driver.env.robot_interface.close_gripper())
    # print("[0]")
    # env.driver.env.robot_interface.close_gripper()
    # print("[1]")
    """
    # env.robot_interface.close_gripper()
    """
    # print("[2]")
    # env.driver.robot_interface.close_gripper()


    # for i in range(50):
    #     action = np.array([0.05, 0, 0, 0])
    #     obs, _, done, _ = env.step(action)
    #     if done:
    #         # break
    #         pass



    # import time
    # env = LiftEnv()

    # prev_time = time.time()
    # open = False
    # while True:
    #     if time.time() - prev_time > 3:
    #         if open:
    #             print("open")
    #             env.step(np.array([0.05, 0, 0, 0, 0, 0, -1]))
    #             open = False
    #         else:
    #             print("close")
    #             env.step(np.array([-0.05, 0, 0, 0, 0, 0, 1]))
    #             open = True
    #         prev_time = time.time()