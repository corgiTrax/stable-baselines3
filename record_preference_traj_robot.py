import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
import argparse
import sys
from moviepy.editor import ImageSequenceClip
sys.path.insert(1, '/home/robot/perls2')
from demos.sawyer_osc_2d import OpSpaceLineXYZ
from real_sawyer_env import RealSawyerReachingEnv

class RealsenseStreamer():
    def __init__(self, device_name=None):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        ## enable device with serial number
        ## 752112070904 (side view)
        ## 832112071449 (overhead view)
        if device_name is None:
            print('using the default camera')
           # self.config.enable_device('832112071449') ## serial number on the camera 752112070904
        elif device_name == 'side_view':
           self.config.enable_device('752112070904')
        elif device_name == 'overhead_view':
           self.config.enable_device('832112071449')
        elif device_name == 'multimod_side':
           self.config.enable_device('617203001978')
        elif device_name == 'multimod_top':
           self.config.enable_device('620201002802')
        else:
           raise NotImplementedError
        if device_name == 'multimod_side' or device_name == 'multimod_top':
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        else:
            self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

        self.align_to_color = rs.align(rs.stream.color)

        # Start streaming
        self.pipe_profile = self.pipeline.start(self.config)

        # Intrinsics & Extrinsics
        frames = self.pipeline.wait_for_frames()
        frames = self.align_to_color.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        self.depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        self.color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        self.depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(
            color_frame.profile)
        self.colorizer = rs.colorizer()

    def capture_rgb(self):
        color_frame = None
        while True:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if color_frame is not None:
                color_image = np.asanyarray(color_frame.get_data())
                break
        return color_image

    def capture_rgbd(self):
        frame_error = True
        while frame_error:
            try:
                frames = self.align_to_color.process(frames)  
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                frame_error = False
            except:
                frames = self.pipeline.wait_for_frames()
                continue
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())
        return color_frame, color_image, depth_frame, depth_image

    def stop_stream(self):
        self.pipeline.stop()

    def show_image(self, image):
        cv2.imshow('img', image)
        cv2.waitKey(0)


def record_traj(
    env,
    txt_save_path = "preference_traj/data",
    img_save_path = "preference_traj/images",
    video_save_path = "preference_traj/videos",
    ):

    realsense_streamer = RealsenseStreamer()
    # for i in range (5):
    #     frames = realsense_streamer.pipeline.wait_for_frames()
    #     color_frame = frames.get_color_frame()
    #     color_img_front = np.asanyarray(color_frame.get_data())
    #     # realsense_streamer.show_image(color_img_front)
    #     cv2.imwrite(f"{i}.jpg", color_img_front)

    os.makedirs(txt_save_path, exist_ok=True)
    txt_file = open(os.path.join(txt_save_path, "time_state_action.txt"), "w")

    os.makedirs(img_save_path, exist_ok=True)
    os.makedirs(video_save_path, exist_ok=True)
    
    # reset environment to get initial state
    state = env.reset()

    test_action = [
        np.array([0.25, 0, 0, 0]),
        np.array([0.25, 0, 0, 0]),
        np.array([0.25, 0, 0, 0]),
        np.array([-0.25, 0, 0, 0]),
        np.array([0, 0.25, 0, 0]),
        np.array([0, 0.25, 0, 0]),
        np.array([0, 0.25, 0, 0]),
        np.array([0 -0.25, 0, 0]),
        np.array([0 -0.25, 0, 0])
    ]

    start_time = time.time()

    frames = []
    for timestep in range(len(test_action)):
        # generate random action
        # action = np.random.uniform(-0.25, 0.25, 4)
        action = test_action[timestep]
        print(f"time {timestep}: action {action}")

        # record to txt
        txt_file.write(
            f"Timestep = {str(timestep)}, Time = {str(time.time()-start_time)} State = {str(state)}, Action = {str(action)}\n"
        )

        # save image
        frame = realsense_streamer.pipeline.wait_for_frames()
        color_frame = frame.get_color_frame()
        color_img_front = cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_RGB2BGR)
        frames.append(color_img_front)
        # cv2.imwrite(f"{img_save_path}/step{timestep}.jpg", color_img_front)

        # take step
        state, _, _, _ = env.step(action)
    
    clip = ImageSequenceClip(frames, fps=10)
    clip.write_videofile(video_save_path+"/traj1.mp4", audio=False)





if __name__ == "__main__":
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

    env = RealSawyerReachingEnv(driver, random_init=False)

    record_traj(env)

