import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), './basic_shapes'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../visualization'))

from typing import Optional

import numpy as np

from basic_shapes_utils import create_box
from data_format_checker import check_pointcloud_format
from label import PlaneLabel
from simple_visualizer import visualize_pc_color


def disturb_position(
    point_cloud: np.ndarray,
    range: Optional[tuple] = (-0.05, 0.05)
) -> np.ndarray:
    '''Add noise to the x, y-component of the point cloud.

    The position of each point cloud is disturbed uniformly
    in the given range.
    '''
    check_pointcloud_format(point_cloud)
    noise_lower = range[0]
    noise_upper = range[1]
    num_points = point_cloud.shape[0]
    for idx in range(num_points):
        random_noise = noise_lower + (noise_upper - noise_lower) * np.random.random()
        point_cloud[idx][0] += random_noise
        point_cloud[idx][1] += random_noise
    return point_cloud


def disturb_orientation(
    point_cloud: np.ndarray,
    range: Optional[tuple] = (-1, 1)
) -> np.ndarray:
    '''Add noise to the orientation of the point cloud.

    The point cloud is tilted in a random direction by an angle sampled 
    uniformly in the given range.

    '''


def disturb_height(
    point_cloud: np.ndarray,
    range: Optional[tuple] = (-0.05, 0.05)
) -> np.ndarray:
    '''Add noise to the z-component of the point cloud.

    The height of random patches of the point cloud is disturbed uniformly
    in the given range.
    '''
    check_pointcloud_format(point_cloud)
    noise_lower = range[0]
    noise_upper = range[1]
    num_points = point_cloud.shape[0]
    for idx in range(num_points):
        random_noise = noise_lower + (noise_upper - noise_lower) * np.random.random()
        point_cloud[idx][2] += random_noise
    return point_cloud


def add_random_clusters(
    point_cloud: np.ndarray
) -> np.ndarray:
    '''Add random clusters to the point cloud.
    '''
    has_label = check_pointcloud_format(point_cloud)
    num_data = point_cloud.shape[0]
    num_outlier_points_max = int(num_data / 50)
    num_clusters = np.random.randint(6)
    if num_clusters != 0:
        for _ in range(num_clusters):
            cur_outlier_points = np.random.randint(num_outlier_points_max)
            if cur_outlier_points != 0:
                cur_mean = np.array([
                    -3 + 6 * np.random.random(),
                    -3 + 6 * np.random.random(),
                    3 * np.random.random()
                ])
                cur_cov = 0.1 * np.random.random() * np.eye(3)
                cur_gaussian_outlier = np.random.multivariate_normal(cur_mean, cur_cov, cur_outlier_points)

                if has_label:
                    cur_gaussian_outlier_label = PlaneLabel.others.label * np.ones((cur_outlier_points, 4))
                    cur_gaussian_outlier_label[:, 0:3] = cur_gaussian_outlier
                    point_cloud = np.append(point_cloud, cur_gaussian_outlier_label, axis=0)
                else:
                    point_cloud = np.append(point_cloud, cur_gaussian_outlier, axis=0)
    return point_cloud


def remove_random_patches(
    point_cloud: np.ndarray
) -> np.ndarray:
    '''Remove random patches of the point cloud.
    '''
    num_data = point_cloud.shape[0]
    num_remove = np.random.randint(0, int(num_data / 3))
    remove_start_idx = np.random.randint(0, num_data)
    if remove_start_idx + num_remove > num_data - 1:
        remove_end_idx = num_data - 1
    else:
        remove_end_idx = remove_start_idx + num_remove
    
    return np.delete(point_cloud, np.s_[remove_start_idx:remove_end_idx], axis=0)


if __name__ == '__main__':

    # test add random outliers
    box = create_box(2.5, 1.283, 1.344, label=True)
    box_with_outliers = add_random_clusters(box)
    color_dict = {
        PlaneLabel.horizontal.label: PlaneLabel.horizontal.color,
        PlaneLabel.vertical.label: PlaneLabel.vertical.color,
        PlaneLabel.others.label: PlaneLabel.others.color
    }
    visualize_pc_color(box_with_outliers, color_dict)

    # test remove random patches
    data_remove_test = np.arange(30).reshape(10, 3)
    print(remove_random_patches(data_remove_test))