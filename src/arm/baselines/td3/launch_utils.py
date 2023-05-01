import logging
from typing import List

import numpy as np
from rlbench.backend.observation import Observation
from rlbench.demo import Demo
from yarr.replay_buffer.prioritized_replay_buffer import \
    PrioritizedReplayBuffer
from yarr.replay_buffer.replay_buffer import ReplayElement, ReplayBuffer
from yarr.replay_buffer.uniform_replay_buffer import UniformReplayBuffer

from get_image.src.arm import demo_loading_utils, utils
from get_image.src.arm.baselines.td3.td3_agent import TD3Agent
from get_image.src.arm.custom_rlbench_env import CustomRLBenchEnv
from get_image.src.arm.network_utils import SiameseNet, CNNAndFcsNet
from get_image.src.arm.preprocess_agent import PreprocessAgent

REWARD_SCALE = 100.0


def create_replay(batch_size: int, timesteps: int, prioritisation: bool,
                  save_dir: str, env: CustomRLBenchEnv):
    observation_elements = env.observation_elements
    replay_class = UniformReplayBuffer
    if prioritisation:
        replay_class = PrioritizedReplayBuffer
    replay_buffer = replay_class(
        save_dir=save_dir,
        batch_size=batch_size,
        timesteps=timesteps,
        replay_capacity=int(1e5),
        action_shape=(8,),
        action_dtype=np.float32,
        reward_shape=(),
        reward_dtype=np.float32,
        update_horizon=1,
        observation_elements=observation_elements,
        extra_replay_elements=[ReplayElement('demo', (), np.bool)]
    )
    return replay_buffer


def _get_action(obs_tp1: Observation):
    quat = utils.normalize_quaternion(obs_tp1.gripper_pose[3:])
    if quat[-1] < 0:
        quat = -quat
    return np.concatenate([obs_tp1.gripper_pose[:3], quat,
                           [float(obs_tp1.gripper_open)]])


def _add_keypoints_to_replay(
        replay: ReplayBuffer,
        inital_obs: Observation,
        demo: Demo,
        env: CustomRLBenchEnv,
        episode_keypoints: List[int]):
    prev_action = None
    obs = inital_obs
    all_actions = []
    for k, keypoint in enumerate(episode_keypoints):
        obs_tp1 = demo[keypoint]
        action = _get_action(obs_tp1)
        all_actions.append(action)
        terminal = (k == len(episode_keypoints) - 1)
        reward = float(terminal) * REWARD_SCALE if terminal else 0
        obs_dict = env.extract_obs(obs, t=k, prev_action=prev_action)
        prev_action = np.copy(action)
        others = {'demo': True}
        others.update(obs_dict)
        timeout = False
        replay.add(action, reward, terminal, timeout, **others)
        obs = obs_tp1  # Set the next obs
    # Final step
    obs_dict_tp1 = env.extract_obs(
        obs_tp1, t=k + 1, prev_action=prev_action)
    replay.add_final(**obs_dict_tp1)
    return all_actions


def fill_replay(replay: ReplayBuffer,
                task: str,
                env: CustomRLBenchEnv,
                num_demos: int,
                demo_augmentation: bool,
                demo_augmentation_every_n: int):
    logging.info('Filling replay with demos...')
    all_actions = []
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
            all_actions.extend(_add_keypoints_to_replay(
                replay, obs, demo, env, episode_keypoints))
    logging.info('Replay filled with demos.')
    return all_actions


def create_agent(camera_name: str,
                 activation: str,
                 action_min_max,
                 image_resolution: list,
                 critic_lr,
                 actor_lr,
                 critic_weight_decay,
                 actor_weight_decay,
                 tau,
                 critic_grad_clip,
                 actor_grad_clip,
                 low_dim_state_len):

    siamese_net = SiameseNet(
        input_channels=[3, 3],
        filters=[16],
        kernel_sizes=[5],
        strides=[1],
        activation=activation,
        norm=None,
    )

    critic_net = CNNAndFcsNet(
        siamese_net=siamese_net,
        input_resolution=image_resolution,
        filters=[32, 32, 32],
        kernel_sizes=[3, 3, 3],
        strides=[2, 2, 2],
        norm=None,
        activation=activation,
        fc_layers=[256, 256, 1],
        low_dim_state_len=low_dim_state_len + 8)

    actor_net = CNNAndFcsNet(
        siamese_net=siamese_net,
        input_resolution=image_resolution,
        filters=[32, 32, 32],
        kernel_sizes=[3, 3, 3],
        strides=[2, 2, 2],
        norm=None,
        activation=activation,
        fc_layers=[256, 256, 1],
        low_dim_state_len=low_dim_state_len)

    td3_agent = TD3Agent(
        critic_network=critic_net,
        actor_network=actor_net,
        action_min_max=action_min_max,
        camera_name=camera_name,
        critic_lr=critic_lr,
        actor_lr=actor_lr,
        critic_weight_decay=critic_weight_decay,
        actor_weight_decay=actor_weight_decay,
        critic_tau=tau,
        critic_grad_clip=critic_grad_clip,
        actor_grad_clip=actor_grad_clip)

    return PreprocessAgent(pose_agent=td3_agent)
