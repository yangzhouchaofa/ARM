#!/usr/bin/env python
import os
import pickle

import cv2
import numpy as np
import rospy
from PIL import Image as pImage
from cv_bridge import CvBridge
from diagnostic_msgs.msg import DiagnosticArray
from dynamic_reconfigure.msg import ConfigDescription
from geometry_msgs.msg import WrenchStamped
from rlbench.demo import Demo
from sensor_msgs.msg import CompressedImage, Image, JointState, CameraInfo
from std_msgs.msg import String

import utils
from tf2_ros import TransformListener, Buffer

example_path = '/home/ylc/ARM/yt_data/tmp'
EPISODES_FOLDER = 'episodes'
EPISODE_FOLDER = 'episode%d'
WRIST_RGB_FOLDER = 'wrist_rgb'
WRIST_DEPTH_FOLDER = 'wrist_depth'
LOW_DIM_PICKLE = 'low_dim_obs.pkl'
IMAGE_FORMAT = '%d.png'
DEPTH_SCALE = 2 ** 24 - 1


class Observation(object):
    """Storage for both visual and low-dimensional observations."""

    def __init__(self,
                 left_shoulder_rgb: np.ndarray,
                 left_shoulder_depth: np.ndarray,
                 left_shoulder_mask: np.ndarray,
                 left_shoulder_point_cloud: np.ndarray,
                 right_shoulder_rgb: np.ndarray,
                 right_shoulder_depth: np.ndarray,
                 right_shoulder_mask: np.ndarray,
                 right_shoulder_point_cloud: np.ndarray,
                 overhead_rgb: np.ndarray,
                 overhead_depth: np.ndarray,
                 overhead_mask: np.ndarray,
                 overhead_point_cloud: np.ndarray,
                 wrist_rgb: np.ndarray,
                 wrist_depth: np.ndarray,
                 wrist_mask: np.ndarray,
                 wrist_point_cloud: np.ndarray,
                 front_rgb: np.ndarray,
                 front_depth: np.ndarray,
                 front_mask: np.ndarray,
                 front_point_cloud: np.ndarray,
                 joint_velocities: np.ndarray,
                 joint_positions: np.ndarray,
                 joint_forces: np.ndarray,
                 gripper_open: float,
                 gripper_pose: np.ndarray,
                 gripper_matrix: np.ndarray,
                 gripper_joint_positions: np.ndarray,
                 gripper_touch_forces: np.ndarray,
                 task_low_dim_state: np.ndarray,
                 misc: dict):
        self.left_shoulder_rgb = left_shoulder_rgb
        self.left_shoulder_depth = left_shoulder_depth
        self.left_shoulder_mask = left_shoulder_mask
        self.left_shoulder_point_cloud = left_shoulder_point_cloud
        self.right_shoulder_rgb = right_shoulder_rgb
        self.right_shoulder_depth = right_shoulder_depth
        self.right_shoulder_mask = right_shoulder_mask
        self.right_shoulder_point_cloud = right_shoulder_point_cloud
        self.overhead_rgb = overhead_rgb
        self.overhead_depth = overhead_depth
        self.overhead_mask = overhead_mask
        self.overhead_point_cloud = overhead_point_cloud
        self.wrist_rgb = wrist_rgb
        self.wrist_depth = wrist_depth
        self.wrist_mask = wrist_mask
        self.wrist_point_cloud = wrist_point_cloud
        self.front_rgb = front_rgb
        self.front_depth = front_depth
        self.front_mask = front_mask
        self.front_point_cloud = front_point_cloud
        self.joint_velocities = joint_velocities
        self.joint_positions = joint_positions
        self.joint_forces = joint_forces
        self.gripper_open = gripper_open
        self.gripper_pose = gripper_pose
        self.gripper_matrix = gripper_matrix
        self.gripper_joint_positions = gripper_joint_positions
        self.gripper_touch_forces = gripper_touch_forces
        self.task_low_dim_state = task_low_dim_state
        self.misc = misc

    def get_low_dim_data(self) -> np.ndarray:
        """Gets a 1D array of all the low-dimensional obseervations.

        :return: 1D array of observations.
        """
        low_dim_data = [] if self.gripper_open is None else [[self.gripper_open]]
        for data in [self.joint_velocities, self.joint_positions,
                     self.joint_forces,
                     self.gripper_pose, self.gripper_joint_positions,
                     self.gripper_touch_forces, self.task_low_dim_state]:
            if data is not None:
                low_dim_data.append(data)
        return np.concatenate(low_dim_data) if len(low_dim_data) > 0 else np.array([])


def main():
    rospy.init_node('get_image_for_arm')
    a = 10
    demo = []
    bridge = CvBridge()
    episodes_path = os.path.join(example_path, EPISODES_FOLDER)
    check_and_make(episodes_path)
    for ex_idx in range(2):
        while a:
            tfBuffer = Buffer()
            listener = TransformListener(tfBuffer)  # can not be delete
            obs = Observation
            wrist_rgb_buffer = rospy.wait_for_message('/camera/color/image_raw', Image)
            wrist_depth_buffer = rospy.wait_for_message('/camera/aligned_depth_to_color/image_raw', Image)
            joint_states = rospy.wait_for_message('/joint_states', JointState)
            # camera_info = rospy.wait_for_message('/camera/color/camera_info', CameraInfo, timeout=None)
            # camera_info = rospy.wait_for_message('/camera/depth/image_rect_raw/compressed/parameter_descriptions', ConfigDescription, timeout=None)
            # parameter_descriptions = rospy.wait_for_message('/camera/stereo_module/auto_exposure_roi/parameter_descriptions', ConfigDescription, timeout=None)
            # diagnostics = rospy.wait_for_message('/diagnostics', DiagnosticArray, timeout=None)
            # Status_Message = rospy.wait_for_message('/camera/realsense2_camera_manager/bond', Status, timeout=None)
            # Status_Message = rospy.wait_for_message('/franka_state_controller/F_ext', WrenchStamped, timeout=None)
            # rospy.loginfo(camera_info)
            trans = tfBuffer.lookup_transform('world', 'panda_hand', rospy.Time())
            gripper_pose = np.array(
                [trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z,
                 trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z,
                 trans.transform.rotation.w])
            rospy.loginfo(gripper_pose)
            joint_positions = joint_states.position[:7]
            gripper_joint_positions = joint_states.position[7:]

            wrist_rgb_cv_image = bridge.imgmsg_to_cv2(wrist_rgb_buffer, 'rgb8')
            wrist_depth_cv_image = bridge.imgmsg_to_cv2(wrist_depth_buffer, '16UC1')

            # resize to (128,128)
            wrist_rgb = cv2.resize(wrist_rgb_cv_image, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
            wrist_depth = cv2.resize(wrist_depth_cv_image, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)

            obs.wrist_rgb = wrist_rgb
            obs.wrist_depth = wrist_depth
            obs.joint_positions = joint_positions
            obs.gripper_joint_positions = gripper_joint_positions

            a = a - 1
            demo.append(obs)
        Demo(demo)
        episode_path = os.path.join(episodes_path, EPISODE_FOLDER % ex_idx)
        save_demo(demo, episode_path)


def save_demo(demo, episode_path):
    wrist_rgb_path = os.path.join(episode_path, WRIST_RGB_FOLDER)
    wrist_depth_path = os.path.join(episode_path, WRIST_DEPTH_FOLDER)

    check_and_make(wrist_rgb_path)
    check_and_make(wrist_depth_path)

    for i, obs in enumerate(demo):
        wrist_rgb = pImage.fromarray(obs.wrist_rgb)
        wrist_depth = pImage.fromarray(obs.wrist_depth)
        wrist_depth = utils.float_array_to_rgb_image(
            obs.wrist_depth/1000, scale_factor=DEPTH_SCALE)

        wrist_rgb.save(os.path.join(wrist_rgb_path, IMAGE_FORMAT % i))
        wrist_depth.save(os.path.join(wrist_depth_path, IMAGE_FORMAT % i))

    with open(os.path.join(episode_path, LOW_DIM_PICKLE), 'wb') as f:
        pickle.dump(demo, f)


def check_and_make(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


if __name__ == '__main__':
    main()
