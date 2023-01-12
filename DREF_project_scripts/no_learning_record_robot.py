import numpy as np
import yaml
import matplotlib.pyplot as plt
import argparse
import sys
import os
import random

sys.path.insert(1, '/home/robot/perls2')
# perls2 modules
from demos.sawyer_osc_2d import OpSpaceLineXYZ
from real_sawyer_env import RealSawyerReachingEnv

def main():

    # read config file
    with open("configs/robosuite/random_record.yaml", "r") as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)

    # initialize feedback.txt file
    experiment_save_dir = config_data['data_save_path']
    os.makedirs(experiment_save_dir, exist_ok=True)
    feedback_file = open(os.path.join(experiment_save_dir, "feedback_file.txt"), "w")


    # initialization
    n_timesteps = 0
    curr_episode_timesteps = 0
    done = False
    total_rewards = 0

    tensorboard_log_dir = config_data["tensorboard_log_dir"]

    rewards = [0]
    total_time = [0]

    np.random.seed(33)


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

    env = RealSawyerReachingEnv(driver)

    np.set_printoptions(threshold=np.inf)

    os.makedirs(tensorboard_log_dir, exist_ok=True)

    while os.path.exists(config_data['data_save_path']):
        config_data['data_save_path'] = "/".join(config_data['data_save_path'].split("/")[:-1]) + '/run_' + str(int(random.random() * 1000000000))


    # run 
    while n_timesteps < config_data['steps']:
        
        while not done:

            print(f"Time {n_timesteps} ({curr_episode_timesteps})")

            # generate random action
            action =  np.random.uniform(-0.25, 0.25, 4)

            # take step
            state, reward, done, info = env.step(action)
            total_rewards += reward

            # record
            feedback_file.write(
                f"Current timestep = {str(n_timesteps)}. State = {str(state)}. Action = {str(action)}. Reward = {str(reward)}\n"
            )
            feedback_file.write(
                f"Curr episode timestep = {str(curr_episode_timesteps)}\n"
            )

            if info['out_of_bounds']:
                feedback_file.write(
                    f"Action out of bounds\n"
                )

            n_timesteps += 1
            curr_episode_timesteps += 1

            rewards.append(total_rewards)
            total_time.append(n_timesteps)

            if n_timesteps >= config_data['steps']:
                break

        state = env.reset()
        done = False
        curr_episode_timesteps = 0

    # print("time", total_time)
    # print("reward", rewards)
    np.savetxt("random_time.txt", np.array(total_time))
    np.savetxt("random_rewards.txt", np.array(rewards))


if __name__ == "__main__":
    main()