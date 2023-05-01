import logging
from typing import List

import numpy as np
from omegaconf import DictConfig
from rlbench.backend.observation import Observation
from rlbench.demo import Demo
from yarr.envs.env import Env
from yarr.replay_buffer.prioritized_replay_buffer import \
    PrioritizedReplayBuffer, ObservationElement
from yarr.replay_buffer.replay_buffer import ReplayElement, ReplayBuffer
from yarr.replay_buffer.uniform_replay_buffer import UniformReplayBuffer

from get_image.src.arm import demo_loading_utils, utils
from get_image.src.arm.c2farm.networks import Qattention3DNet
from get_image.src.arm.c2farm.qattention_agent import QAttentionAgent
from get_image.src.arm.c2farm.qattention_stack_agent import QAttentionStackAgent
from get_image.src.arm.custom_rlbench_env import CustomRLBenchEnv
from get_image.src.arm.lpr.combined_agent import CombinedAgent
from get_image.src.arm.lpr.const import UNKNOWN_COLLISION
from get_image.src.arm.lpr.networks import PointNet, DenseNet
from get_image.src.arm.lpr.trajectory_agent import TrajectoryAgent, \
    TrajectoryWithActorAgent
from get_image.src.arm.preprocess_agent import PreprocessAgent

REWARD_SCALE = 100.0


def create_replay(batch_size: int, timesteps: int, prioritisation: bool,
                  save_dir: str, cameras: list, env: Env,
                  voxel_sizes, trajectory_points: int, trajectory_mode: str, replay_size=1e5):

    if trajectory_mode not in ['joint', 'pose', 'joint+pose']:
        raise ValueError('Invalid trajectory mode', trajectory_mode)

    trans_indicies_size = 3 * len(voxel_sizes)
    rot_and_grip_indicies_size = (3 + 1)
    traj_size = trajectory_points * (14 if trajectory_mode == 'joint+pose' else 7)

    observation_elements = env.observation_elements
    for cname in cameras:
        observation_elements.append(
            ObservationElement('%s_pixel_coord' % cname, (2,), np.int32))
    observation_elements.extend([
        ReplayElement('trans_action_indicies', (trans_indicies_size,),
                      np.int32),
        ReplayElement('rot_grip_action_indicies', (rot_and_grip_indicies_size,),
                      np.int32),
        ReplayElement('trajectory', (traj_size + 3,), np.float32),
    ])

    for depth in range(len(voxel_sizes)):
        observation_elements.append(
            ReplayElement('attention_coordinate_layer_%d' % depth, (3,), np.float32)
        )

    extra_replay_elements = [
        ReplayElement('demo', (), np.bool),
        ReplayElement('ep_success', (), np.bool),
        ReplayElement('linspace', (trajectory_points, 7), np.float32)  # used for pi traj
    ]

    replay_class = UniformReplayBuffer
    if prioritisation:
        replay_class = PrioritizedReplayBuffer
    replay_buffer = replay_class(
        save_dir=save_dir,
        batch_size=batch_size,
        timesteps=timesteps,
        replay_capacity=int(replay_size),
        action_shape=(traj_size + 1,),
        action_dtype=np.float32,
        reward_shape=(),
        reward_dtype=np.float32,
        update_horizon=1,
        observation_elements=observation_elements,
        extra_replay_elements=extra_replay_elements
    )
    return replay_buffer


def _get_action(
        obs_tp1: Observation,
        rlbench_scene_bounds: List[float],   # AKA: DEPTH0_BOUNDS
        voxel_sizes: List[int],
        bounds_offset: List[float],
        rotation_resolution: int,
        crop_augmentation: bool):
    quat = utils.normalize_quaternion(obs_tp1.gripper_pose[3:])
    if quat[-1] < 0:
        quat = -quat
    disc_rot = utils.quaternion_to_discrete_euler(quat, rotation_resolution)
    assert len(bounds_offset) == len(voxel_sizes) -1
    attention_coordinate = obs_tp1.gripper_pose[:3]
    trans_indicies, attention_coordinates = [], []
    bounds = np.array(rlbench_scene_bounds)
    for depth, vox_size in enumerate(voxel_sizes):
        if depth > 0:
            if crop_augmentation:
                shift = bounds_offset[depth - 1] * 0.75
                attention_coordinate += np.random.uniform(-shift, shift, size=(3,))
            bounds = np.concatenate([attention_coordinate - bounds_offset[depth - 1],
                                     attention_coordinate + bounds_offset[depth - 1]])
        index = utils.point_to_voxel_index(
            obs_tp1.gripper_pose[:3], vox_size, bounds)
        trans_indicies.extend(index.tolist())
        res = (bounds[3:] - bounds[:3]) / vox_size
        attention_coordinate = bounds[:3] + res * index
        attention_coordinates.append(attention_coordinate)

    rot_and_grip_indicies = disc_rot.tolist()
    grip = float(obs_tp1.gripper_open)
    rot_and_grip_indicies.extend([int(obs_tp1.gripper_open)])
    return trans_indicies, rot_and_grip_indicies, np.concatenate(
        [obs_tp1.gripper_pose, np.array([grip])]), attention_coordinates


def _interp_poses(demo, prev_keypoint, keypoint, trajectory_points):
    cart_poses = np.array(
        [demo[i].gripper_pose for i in range(prev_keypoint, keypoint + 1)])
    cfgs = np.array(
        [demo[i].joint_positions for i in range(prev_keypoint, keypoint + 1)])
    sample_points = np.linspace(0, 1, trajectory_points)
    sampled_cart_poses, sampled_cfgs = [], []
    for i in sample_points:
        p = cart_poses[int(i * (len(cart_poses) - 1))]
        cfg = cfgs[int(i * (len(cfgs) - 1))]
        quat = utils.normalize_quaternion(p[3:])
        if quat[-1] < 0:
            quat = -quat
            p[3:] = quat
        sampled_cart_poses.append(p)
        sampled_cfgs.append(cfg)
    return np.array(sampled_cart_poses), np.array(sampled_cfgs)


def _add_keypoints_to_replay(
        replay: ReplayBuffer,
        inital_obs: Observation,
        demo: Demo,
        env: CustomRLBenchEnv,
        episode_keypoints: List[int],
        cameras: List[str],
        rlbench_scene_bounds: List[float],   # AKA: DEPTH0_BOUNDS
        voxel_sizes: List[int],
        bounds_offset: List[float],
        rotation_resolution: int,
        crop_augmentation: bool,
        initial_index: int,
        trajectory_points: int,
        trajectory_mode: str):
    prev_action = None
    obs = inital_obs
    prev_keypoint = initial_index
    for k, keypoint in enumerate(episode_keypoints):
        obs_tp1 = demo[keypoint]
        trans_indicies, rot_grip_indicies, action, attention_coordinates = _get_action(
            obs_tp1, rlbench_scene_bounds, voxel_sizes, bounds_offset,
            rotation_resolution, crop_augmentation)

        cart_poses, cfgs = _interp_poses(
            demo, prev_keypoint, keypoint, trajectory_points)

        terminal = (k == len(episode_keypoints) - 1)
        reward = float(terminal) * REWARD_SCALE if terminal else 0

        obs_dict = env.extract_obs(obs, t=k, prev_action=prev_action)
        prev_action = np.copy(action)

        st = np.linspace(demo[prev_keypoint].gripper_pose, demo[keypoint].gripper_pose, trajectory_points)
        st[:, 3:] = utils.normalize_quaternion(st[:, 3:])
        others = {'demo': True,
                  'ep_success': True,
                  'linspace': st}
        final_obs = {
            'trajectory': np.concatenate([
                cart_poses.flatten() if trajectory_mode == 'pose'
                else cfgs.flatten(), UNKNOWN_COLLISION], 0),
            'trans_action_indicies': trans_indicies,
            'rot_grip_action_indicies': rot_grip_indicies,
        }

        for depth in range(len(voxel_sizes)):
            final_obs['attention_coordinate_layer_%d' % depth] = \
                attention_coordinates[depth]
        for name in cameras:
            px, py = utils.point_to_pixel_index(
                obs_tp1.gripper_pose[:3],
                obs_tp1.misc['%s_camera_extrinsics' % name],
                obs_tp1.misc['%s_camera_intrinsics' % name])
            final_obs['%s_pixel_coord' % name] = [py, px]
        others.update(final_obs)
        others.update(obs_dict)
        timeout = False
        a = np.concatenate([cfgs.flatten(), action[-1:]])
        replay.add(a, reward, terminal, timeout, **others)
        obs = obs_tp1  # Set the next obs

        prev_keypoint = keypoint

    # Final step
    obs_dict_tp1 = env.extract_obs(
        obs_tp1, t=k + 1, prev_action=prev_action)
    obs_dict_tp1.pop('wrist_world_to_cam', None)
    obs_dict_tp1.update(final_obs)
    replay.add_final(**obs_dict_tp1)


def fill_replay(replay: ReplayBuffer,
                task: str,
                env: CustomRLBenchEnv,
                num_demos: int,
                demo_augmentation: bool,
                demo_augmentation_every_n: int,
                cameras: List[str],
                rlbench_scene_bounds: List[float],  # AKA: DEPTH0_BOUNDS
                voxel_sizes: List[int],
                bounds_offset: List[float],
                rotation_resolution: int,
                crop_augmentation: bool,
                trajectory_points: int,
                trajectory_mode: str):

    logging.info('Filling replay with demos...')
    for d_idx in range(num_demos):
        demo = env.env.get_demos(
            task, 1, variation_number=0, random_selection=False,
            from_episode_number=d_idx)[0]
        episode_keypoints = demo_loading_utils.keypoint_discovery(demo)

        for i in range(len(demo) - 1):
            if not demo_augmentation and i > 0:
                break
            if i % demo_augmentation_every_n != 0:
                continue
            obs = demo[i]
            # If our starting point is past one of the keypoints, then remove it
            while len(episode_keypoints) > 0 and i >= episode_keypoints[0]:
                episode_keypoints = episode_keypoints[1:]
            if len(episode_keypoints) == 0:
                break
            if (episode_keypoints[0] - i) < trajectory_points:
                continue
            _add_keypoints_to_replay(
                replay, obs, demo, env, episode_keypoints, cameras,
                rlbench_scene_bounds, voxel_sizes, bounds_offset,
                rotation_resolution, crop_augmentation, i, trajectory_points,
                trajectory_mode)
    logging.info('Replay filled with demos.')


def create_agent(cfg: DictConfig, env, depth_0bounds, cam_resolution,
                 trajectory_point_noise: float, trajectory_points: int,
                 trajectory_mode: str, trajectory_samples: int):
    VOXEL_FEATS = 3
    LATENT_SIZE = 64
    depth_0bounds = depth_0bounds
    cam_resolution = cam_resolution

    include_prev_layer = False

    num_rotation_classes = int(360. // cfg.method.rotation_resolution)
    qattention_agents = []
    for depth, vox_size in enumerate(cfg.method.voxel_sizes):
        last = depth == len(cfg.method.voxel_sizes) - 1
        unet3d = Qattention3DNet(
            in_channels=VOXEL_FEATS + 3 + 1 + 3,
            out_channels=1,
            voxel_size=vox_size,
            out_dense=((num_rotation_classes * 3) + 2) if last else 0,
            kernels=LATENT_SIZE,
            norm=None if 'None' in cfg.method.norm else cfg.method.norm,
            dense_feats=128,
            activation=cfg.method.activation,
            low_dim_size=env.low_dim_state_len,
            include_prev_layer=include_prev_layer and depth > 0)


        qattention_agent = QAttentionAgent(
            layer=depth,
            coordinate_bounds=depth_0bounds,
            unet3d=unet3d,
            camera_names=cfg.rlbench.cameras,
            voxel_size=vox_size,
            bounds_offset=cfg.method.bounds_offset[depth - 1] if depth > 0 else None,
            image_crop_size=cfg.method.image_crop_size,
            tau=cfg.method.tau,
            lr=cfg.method.lr,
            lambda_trans_qreg=cfg.method.lambda_trans_qreg,
            lambda_rot_qreg=cfg.method.lambda_rot_qreg,
            include_low_dim_state=True,
            image_resolution=cam_resolution,
            batch_size=cfg.replay.batch_size,
            voxel_feature_size=VOXEL_FEATS,
            exploration_strategy=cfg.method.exploration_strategy,
            lambda_weight_l2=cfg.method.lambda_weight_l2,
            num_rotation_classes=num_rotation_classes,
            rotation_resolution=cfg.method.rotation_resolution,
            grad_clip=0.01,
            gamma=0.99
        )
        qattention_agents.append(qattention_agent)

    dnet = PointNet(
        input_dims=14 if trajectory_mode == 'joint+pose' else 7,
        extra_dims=3+1+7)

    traj_agent_class = TrajectoryAgent
    traj_agent_kwargs = dict(
        network=dnet,
        tau=cfg.method.trajectory_tau,
        lr=cfg.method.trajectory_lr,
        grad_clip=0.1,
        gamma=0.99
    )

    if cfg.method.learn_trajectory_pi:
        traj_agent_class = TrajectoryWithActorAgent
        anet = DenseNet(
            input_size=3 + 1 + 7 + 7,  #+7 is for NBP
            fc_layers=[128, 128, 7*trajectory_points])
        traj_agent_kwargs['actor_network'] = anet

    traj_agent = traj_agent_class(**traj_agent_kwargs)

    rotation_agent = QAttentionStackAgent(
        qattention_agents=qattention_agents,
        rotation_resolution=cfg.method.rotation_resolution,
        camera_names=cfg.rlbench.cameras,
    )

    combined = CombinedAgent(
        traj_agent, rotation_agent, env, trajectory_point_noise,
        trajectory_points, trajectory_mode, trajectory_samples,
        cfg.method.learn_trajectory_pi)
    preprocess_agent = PreprocessAgent(combined)
    return preprocess_agent
