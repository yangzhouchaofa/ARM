import os
import pickle

import open3d as o3d
import numpy as np
from pypcd import pypcd
from PIL import Image
from matplotlib import pyplot as plt

from get_image.src.utils import image_to_float_array
example_path1 ='/home/ylc/ARM/yt_data/real_demo/2/my_reach_target/variation0/episodes/episode0/wrist_depth'
example_path ='/home/ylc/ARM/yt_data/real_demo/2/my_reach_target/variation0/episodes/episode0/wrist_rgb'
example_path2 ='/home/ylc/ARM/yt_data/real_demo/2/my_reach_target/variation0/episodes/episode0'
# example_path1 ='/home/ylc/ARM/yt_data/sim_demo/my_reach_target/variation0/episodes/episode0/wrist_depth'
# example_path ='/home/ylc/ARM/yt_data/sim_demo/my_reach_target/variation0/episodes/episode0/wrist_rgb'
# example_path2 ='/home/ylc/ARM/yt_data/sim_demo/my_reach_target/variation0/episodes/episode0'
save_npy_path = '/home/ylc/ARM/yt_data/pcd_npy'
save_pcd_path = '/home/ylc/ARM/yt_data/pcd'
LOW_DIM_PICKLE = 'low_dim_obs.pkl'
IMAGE_FORMAT = '%d.png'
pcd_format = '%d.pcd'
npy_format = '%d.npy'



def _resize_if_needed(image, size):
    if image.size[0] != size[0] or image.size[1] != size[1]:
        image = image.resize(size)
    return image


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


with open(os.path.join(example_path2, LOW_DIM_PICKLE), 'rb') as c:
    datac = pickle.load(c)

# for i in range(len(datac._observations)):
    i = 0
    a = datac._observations[i].misc['wrist_camera_extrinsics']
    a = np.array(a)
    b = datac._observations[i].misc['wrist_camera_intrinsics']
    b = np.array(b)


    # my red demo
    color_raw = Image.open(os.path.join(example_path, IMAGE_FORMAT % i))
    depth = Image.open(os.path.join(example_path1, IMAGE_FORMAT % i))

    wrist_depth = image_to_float_array(depth, 2 ** 24 - 1)
    color_raw = np.array(color_raw) / 255

    upc = _create_uniform_pixel_coords_image(wrist_depth.shape)
    pc = upc * np.expand_dims(wrist_depth, -1)
    C = np.expand_dims(a[:3, 3], 0).T
    R = a[:3, :3]
    R_inv = R.T  # inverse of rot matrix is transpose
    R_inv_C = np.matmul(R_inv, C)
    extrinsics = np.concatenate((R_inv, -R_inv_C), -1)
    cam_proj_mat = np.matmul(b, extrinsics)
    cam_proj_mat_homo = np.concatenate([cam_proj_mat, [np.array([0, 0, 0, 1])]])
    cam_proj_mat_inv = np.linalg.inv(cam_proj_mat_homo)[0:3]
    world_coords_homo = np.expand_dims(_pixel_to_world_coords(pc, cam_proj_mat_inv), 0)
    wrist_point_cloud = world_coords_homo[..., :-1][0]
    wrist_point_cloud = wrist_point_cloud.reshape(128 * 128, 3)
    color_raw = color_raw.reshape(128 * 128, 3)

    a = [0, -0.3, 0, 1, 0.7, 1]
    k = 0
    for point in wrist_point_cloud:
        k = k + 1
        if point[0] >= a[3] or point[0] <= a[0] \
                or point[1] >= a[4] or point[1] <= a[1] \
                or point[2] >= a[5] or a[2] >= point[2]:
            k = k - 1
            wrist_point_cloud = np.delete(wrist_point_cloud, k, axis=0)
            color_raw = np.delete(color_raw, k, axis=0)

    point = [[0, 0, 0]]
    wrist_point_cloud = np.concatenate((wrist_point_cloud, point), axis=0)
    color_raw = np.concatenate((color_raw, point), axis=0)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(wrist_point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(color_raw)
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.run()

    ################################################################################################################
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # pcd = o3d.geometry.PointCloud()
    # a = datac['pcd'].cpu().numpy()
    # pcd.points = o3d.utility.Vector3dVector(a.reshape(-1, 3))
    # pcd.paint_uniform_color([0, 1, 0])
    #
    # vis.add_geometry(pcd)
    # vis.poll_events()
    # vis.update_renderer()
    # vis.run()
    ####################################################################################################333
    print('执行体素化点云')
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1/16)
    print("正在可视化体素...")
    o3d.visualization.draw_geometries([voxel_grid])

    # print("->正在保存点云")
    o3d.io.write_point_cloud(os.path.join(save_pcd_path, pcd_format % i), pcd, True)
    cloud2 = pypcd.PointCloud.from_path(os.path.join(save_pcd_path, pcd_format % i))
    new2 = cloud2.pc_data.copy()
    npy = np.array([list(new2) for new2 in new2])
    np.save(os.path.join(save_npy_path, npy_format % i), npy)
