#!/usr/bin/env python
import os
import pickle
from time import time
from typing import Callable

import cv2
import numpy as np
import rospy
import tf
from PIL import Image as pImage
from actionlib_msgs.msg import GoalStatusArray
from cv_bridge import CvBridge
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import WrenchStamped
from realsense2_camera.msg import Extrinsics
from rlbench.backend.scene import Scene
from rlbench.backend import observation
from rlbench.demo import Demo
from sensor_msgs.msg import CompressedImage, Image, JointState, CameraInfo
from std_msgs.msg import String
from rlbench.backend.observation import Observation
from rlbench.observation_config import ObservationConfig, CameraConfig
from tf2_geometry_msgs import tf2_geometry_msgs
from tf2_msgs.msg import TFMessage
from tf2_ros import TransformListener, Buffer
from tqdm import tqdm

import utils

EPISODE_FOLDER = 'episode9'
example_path = '/home/ylc/ARM/yt_data/real_demo/2/my_reach_target/variation0'
EPISODES_FOLDER = 'episodes'
WRIST_RGB_FOLDER = 'wrist_rgb'
WRIST_DEPTH_FOLDER = 'wrist_depth'
LOW_DIM_PICKLE = 'low_dim_obs.pkl'
IMAGE_FORMAT = '%d.png'
# wrist_camera
NEAR = 0
FAR = 1

DEPTH_SCALE = 2 ** 24 - 1
bridge = CvBridge()


class My_Sence(Scene):
    def __init__(self,
                 obs_config: ObservationConfig = ObservationConfig(),
                 robot_setup: str = 'panda'):
        self.robot_setup = robot_setup
        self.task = None
        self._obs_config = obs_config
        self._initial_task_state = None
        self._has_init_task = self._has_init_episode = False
        self._variation_index = 0
        self.task_low_dim_state = None
        self._step_callback = None

    def get_observation(self) -> Observation:
        tfBuffer = Buffer()
        listener = TransformListener(tfBuffer)  # can not be delete

        wrist_rgb_buffer = rospy.wait_for_message('/camera/color/image_raw', Image)
        wrist_depth_buffer = rospy.wait_for_message('/camera/aligned_depth_to_color/image_raw', Image)
        joint_states = rospy.wait_for_message('/joint_states', JointState)
        camera_info_depth_to_color = rospy.wait_for_message('/camera/aligned_depth_to_color/camera_info', CameraInfo)
        # camera_info_extrinsics = rospy.wait_for_message('/camera/extrinsics/depth_to_color', Extrinsics)
        F_ext = rospy.wait_for_message('/franka_state_controller/F_ext', WrenchStamped)
        trans = tfBuffer.lookup_transform('world', 'panda_NE', rospy.Time())
        trans_depth = tfBuffer.lookup_transform('world', 'camera_depth_optical_frame', rospy.Time())

        gripper_matrix = tf_to_matrix(trans)
        wrist_camera_extrinsics = tf_to_matrix(trans_depth)
        gripper_pose = np.array(
            [trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z,
             trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z,
             trans.transform.rotation.w])

        wrist_rgb_cv_image = bridge.imgmsg_to_cv2(wrist_rgb_buffer, 'rgb8')
        wrist_depth_cv_image = bridge.imgmsg_to_cv2(wrist_depth_buffer, desired_encoding='passthrough')
        # resize(640,480) to (128,128)
        wrist_rgb = cv2.resize(wrist_rgb_cv_image, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
        wrist_depth = cv2.resize(wrist_depth_cv_image, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)

        joint_positions = joint_states.position[:7]
        gripper_joint_positions = joint_states.position[7:]
        joint_velocities = joint_states.velocity[:7]
        joint_forces = joint_states.effort[:7]
        gripper_touch_forces = [F_ext.wrench.force.x, F_ext.wrench.force.y, F_ext.wrench.force.z,
                                F_ext.wrench.torque.x, F_ext.wrench.torque.y, F_ext.wrench.torque.z]

        w = 128 / 640
        h = 128 / 480
        wrist_camera_intrinsics = np.array(camera_info_depth_to_color.K)
        wrist_camera_intrinsics[0] = wrist_camera_intrinsics[0] * w
        wrist_camera_intrinsics[2] = wrist_camera_intrinsics[2] * w
        wrist_camera_intrinsics[4] = wrist_camera_intrinsics[4] * h
        wrist_camera_intrinsics[5] = wrist_camera_intrinsics[5] * h
        misc = {'wrist_camera_near': NEAR,
                'wrist_camera_far': FAR,
                'wrist_camera_extrinsics': wrist_camera_extrinsics,
                'wrist_camera_intrinsics': wrist_camera_intrinsics.reshape(3, 3)}

        obs = Observation(
            left_shoulder_rgb=None,
            left_shoulder_depth=None,
            left_shoulder_point_cloud=None,
            right_shoulder_rgb=None,
            right_shoulder_depth=None,
            right_shoulder_point_cloud=None,
            overhead_rgb=None,
            overhead_depth=None,
            overhead_point_cloud=None,
            wrist_rgb=wrist_rgb,
            wrist_depth=wrist_depth,
            wrist_point_cloud=None,
            front_rgb=None,
            front_depth=None,
            front_point_cloud=None,
            left_shoulder_mask=None,
            right_shoulder_mask=None,
            overhead_mask=None,
            wrist_mask=None,
            front_mask=None,
            joint_velocities=joint_velocities,
            joint_positions=joint_positions,
            joint_forces=joint_forces,
            gripper_open=(1.0 if gripper_joint_positions[0] > 0.030 else 0.0),  # 1 means open, whilst 0 means closed.
            gripper_pose=gripper_pose,
            gripper_matrix=gripper_matrix,
            gripper_touch_forces=gripper_touch_forces,
            gripper_joint_positions=gripper_joint_positions,
            task_low_dim_state=self.task_low_dim_state,
            misc=misc)

        return obs

    def _demo_record_step(self, demo_list, record, func):
        if record:
            # demo_list.append(self.get_observation())
            demo_list.insert(0, self.get_observation())
        if func is not None:
            func(self.get_observation())

    def get_demo(self, record: bool = True,
                 callable_each_step: Callable[[Observation], None] = None,
                 randomly_place: bool = True) -> Demo:

        demo = []

        if record:
            self.task_low_dim_state = self.get_observation().gripper_pose[:3]
            demo.append(self.get_observation())

        done = False
        while not done:
            self._demo_record_step(demo, record, callable_each_step)
            msg = rospy.wait_for_message('/move_group/status', GoalStatusArray)
            if len(msg.status_list) == 2 and msg.status_list[1].text == "This goal has been accepted by the simple action server":
                done = True
            rospy.loginfo('done:%d', done)
            # done = True
        return Demo(demo)


def main():
    rospy.init_node('dataset')
    my_sence = My_Sence()
    episodes_path = os.path.join(example_path, EPISODES_FOLDER)
    check_and_make(episodes_path)
    episode_path = os.path.join(episodes_path, EPISODE_FOLDER)
    demo = my_sence.get_demo(record=True)
    save_demo(demo, episode_path=episode_path)


def save_demo(demo, episode_path):
    wrist_rgb_path = os.path.join(episode_path, WRIST_RGB_FOLDER)
    wrist_depth_path = os.path.join(episode_path, WRIST_DEPTH_FOLDER)

    check_and_make(wrist_rgb_path)
    check_and_make(wrist_depth_path)

    for i, obs in tqdm(enumerate(demo)):
        wrist_rgb = pImage.fromarray(obs.wrist_rgb)
        wrist_depth = utils.uint16_array_to_rgb_image(
            obs.wrist_depth, scale_factor=DEPTH_SCALE)

        wrist_rgb.save(os.path.join(wrist_rgb_path, IMAGE_FORMAT % i))
        wrist_depth.save(os.path.join(wrist_depth_path, IMAGE_FORMAT % i))

        # We save the images separately, so set these to None for pickling.
        obs.wrist_rgb = None
        obs.wrist_depth = None

    # Save the low-dimension data
    with open(os.path.join(episode_path, LOW_DIM_PICKLE), 'wb') as f:
        pickle.dump(demo, f)


def check_and_make(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def tf_to_matrix(trans):
    matrix = tf2_geometry_msgs.transform_to_kdl(trans)

    matrix = np.array(
        [[matrix.M[0, 0], matrix.M[0, 1], matrix.M[0, 2], matrix.p.x()],
         [matrix.M[1, 0], matrix.M[1, 1], matrix.M[1, 2], matrix.p.y()],
         [matrix.M[2, 0], matrix.M[2, 1], matrix.M[2, 2], matrix.p.z()],
         [0, 0, 0, 1]])

    return matrix


if __name__ == '__main__':
    main()

