#!/usr/bin/env python
import os
import pickle
from time import time
from typing import Callable
from std_msgs.msg import Float64
import cv2
import numpy as np
import rospy
import tf
from PIL import Image as pImage
from actionlib_msgs.msg import GoalStatusArray
from cv_bridge import CvBridge
from diagnostic_msgs.msg import DiagnosticArray
# from dynamic_reconfigure.msg import ConfigDescription
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

from get_image.src.move_start import move_to_start
from vgn.utils.panda_control import PandaCommander
from vgn.utils.transform import Transform, Rotation
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import utils

# wrist_camera
NEAR = 0
FAR = 1
boundary_max = [1, 0.7, 1]
boundary_min = [0, -0.3, 0]
DEPTH_SCALE = 2 ** 24 - 1
bridge = CvBridge()


class My_Run_Sence(Scene):
    def __init__(self,
                 obs_config: ObservationConfig = ObservationConfig(),
                 robot_setup: str = 'panda'):
        self.robot_setup = robot_setup
        self.task = None
        self._obs_config = obs_config
        self._initial_task_state = None
        self._has_init_task = self._has_init_episode = False
        self._variation_index = 0

        self._step_callback = None

    def get_observation(self) -> Observation:
        tfBuffer = Buffer()
        listener = TransformListener(tfBuffer)  # can not be delete

        wrist_rgb_buffer = rospy.wait_for_message('/camera/color/image_raw', Image)
        wrist_depth_buffer = rospy.wait_for_message('/camera/aligned_depth_to_color/image_raw', Image)
        joint_states = rospy.wait_for_message('/joint_states', JointState)
        camera_info_depth_to_color = rospy.wait_for_message('/camera/aligned_depth_to_color/camera_info', CameraInfo)
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
        # resize to (128,128)
        wrist_rgb = cv2.resize(wrist_rgb_cv_image, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
        wrist_depth = cv2.resize(wrist_depth_cv_image, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)

        joint_positions = joint_states.position[:7]
        gripper_joint_positions = joint_states.position[7:]
        joint_velocities = joint_states.velocity[:7]

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

        wrist_depth = wrist_depth / 1000
        depth_m = NEAR + wrist_depth * (FAR - NEAR)

        upc = _create_uniform_pixel_coords_image(depth_m.shape)
        pc = upc * np.expand_dims(depth_m, -1)
        C = np.expand_dims(misc['wrist_camera_extrinsics'][:3, 3], 0).T
        R = misc['wrist_camera_extrinsics'][:3, :3]
        R_inv = R.T  # inverse of rot matrix is transpose
        R_inv_C = np.matmul(R_inv, C)
        extrinsics = np.concatenate((R_inv, -R_inv_C), -1)
        cam_proj_mat = np.matmul(misc['wrist_camera_intrinsics'], extrinsics)
        cam_proj_mat_homo = np.concatenate([cam_proj_mat, [np.array([0, 0, 0, 1])]])
        cam_proj_mat_inv = np.linalg.inv(cam_proj_mat_homo)[0:3]
        world_coords_homo = np.expand_dims(_pixel_to_world_coords(pc, cam_proj_mat_inv), 0)
        wrist_point_cloud = world_coords_homo[..., :-1][0]

        # # save the pcd
        # wrist_point_cloud = wrist_point_cloud.reshape(128 * 128, 3)
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(wrist_point_cloud)
        # o3d.io.write_point_cloud("/media/ylc/YETONG/111111.pcd",pcd, True)

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
            wrist_depth=None,
            wrist_point_cloud=wrist_point_cloud,
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
            joint_forces=None,
            gripper_open=0.0,
            gripper_pose=gripper_pose,
            gripper_matrix=gripper_matrix,
            gripper_touch_forces=None,
            gripper_joint_positions=gripper_joint_positions,
            task_low_dim_state=None,
            misc=misc)

        return obs

    def _demo_record_step(self, demo_list, record, func):
        if record:
            # demo_list.append(self.get_observation())
            demo_list.insert(0, self.get_observation())
        if func is not None:
            func(self.get_observation())

    def step(self, action):
        T_tool0_tcp = Transform.from_dict({"rotation": [0, 0, -0.382, 0.924], "translation": [0, 0, 0.103399999]})
        T_tcp_tool0 = T_tool0_tcp.inverse()
        rospy.loginfo(action)
        # r = R.from_quat(np.array([- 9.69540444e-01, 2.39419397e-01, 5.02147978e-02, - 1.19379640e-02]))
        r = R.from_quat(np.array(action[3:7]))
        T = Transform(r, np.array(action[:3]))
        if (boundary_min[0] <= action[0] <= boundary_max[0]) and \
                (boundary_min[1] <= action[1] <= boundary_max[1]) and \
                (boundary_min[2] <= action[2] <= boundary_max[2]):
            panda = PandaCommander()
            success, plan = panda.goto_pose(T * T_tcp_tool0, velocity_scaling=0.1)
            #############################################################################################################
            rospy.loginfo(len(plan.joint_trajectory.points))
            rospy.loginfo("success=%d", success)
            assert success is True
            if len(plan.joint_trajectory.points) == 0:
                move_to_start()
                reward = -1
                terminal = True
            else:
                if action[7] == 0:
                    panda.grasp(width=0.0, force=20.0)
                else:
                    panda.move_gripper(0.08)
                ros_reward = rospy.wait_for_message('reward', Float64)
                reward = ros_reward.data
                rospy.loginfo('reward = %f', reward)
                ros_reward = rospy.wait_for_message('reward', Float64)
                terminal = ros_reward.data
                rospy.loginfo('terminal = %f', reward)

            #############################################################################################################


            # terminal = False
            obs = self.get_observation()
            return obs, reward, terminal
        else:
            reward = -1
            rospy.loginfo('reward = %f', reward)
            terminal = True
            obs = self.get_observation()
            return obs, reward, terminal


def _create_uniform_pixel_coords_image(resolution: np.ndarray):
    pixel_x_coords = np.reshape(
        np.tile(np.arange(resolution[1]), [resolution[0]]),
        (resolution[0], resolution[1], 1)).astype(np.float32)
    pixel_y_coords = np.reshape(
        np.tile(np.arange(resolution[0]), [resolution[1]]),
        (resolution[1], resolution[0], 1)).astype(np.float32)
    pixel_y_coords = np.transpose(pixel_y_coords, (1, 0, 2))
    uniform_pixel_coords = np.concatenate(
        (pixel_x_coords, pixel_y_coords, np.ones_like(pixel_x_coords)), -1)
    return uniform_pixel_coords


def _pixel_to_world_coords(pixel_coords, cam_proj_mat_inv):
    h, w = pixel_coords.shape[:2]
    pixel_coords = np.concatenate(
        [pixel_coords, np.ones((h, w, 1))], -1)
    world_coords = _transform(pixel_coords, cam_proj_mat_inv)
    world_coords_homo = np.concatenate(
        [world_coords, np.ones((h, w, 1))], axis=-1)
    return world_coords_homo


def _transform(coords, trans):
    h, w = coords.shape[:2]
    coords = np.reshape(coords, (h * w, -1))
    coords = np.transpose(coords, (1, 0))
    transformed_coords_vector = np.matmul(trans, coords)
    transformed_coords_vector = np.transpose(
        transformed_coords_vector, (1, 0))
    return np.reshape(transformed_coords_vector,
                      (h, w, -1))


def tf_to_matrix(trans):
    matrix = tf2_geometry_msgs.transform_to_kdl(trans)

    matrix = np.array(
        [[matrix.M[0, 0], matrix.M[0, 1], matrix.M[0, 2], matrix.p.x()],
         [matrix.M[1, 0], matrix.M[1, 1], matrix.M[1, 2], matrix.p.y()],
         [matrix.M[2, 0], matrix.M[2, 1], matrix.M[2, 2], matrix.p.z()],
         [0, 0, 0, 1]])

    return matrix
