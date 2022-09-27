import os
import sys

cur_path = os.path.dirname(__file__)
sys.path.append(os.path.join(cur_path, '../data/basic_shapes'))

from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from basic_shapes_utils import PlaneLabel, create_stairs


def visualize_pc_no_color(
    point_cloud: np.ndarray, 
    scale: Optional[tuple] = (1, 1, 1)
) -> None:
    '''A simple visualizer for point clouds.

    Paras
    -----
    point_cloud: {list, np.ndarray} with shape (num_points, 3)
        The point cloud to visualize. The columns represent the xyz coordinate 
        of each points, respectively
    scale: tuple
        # TODO.
    '''
    assert type(point_cloud) == list or \
        type(point_cloud) == np.ndarray
    if type(point_cloud) == np.ndarray:
        assert len(point_cloud.shape) == 2
        assert point_cloud.shape[1] == 3
    else:
        # TODO: check format
        point_cloud = np.array(point_cloud)

    ax = plt.figure().add_subplot(111, projection='3d')
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2])
    plt.gca().set_box_aspect(scale)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def visualize_pc_color(
    point_cloud_with_label: Union[list, np.ndarray],
    color_dict: dict,
    scale: Optional[tuple] = (1, 1, 1)
) -> None:
    '''A simple visualizer for point clouds and labels.

    Paras
    -----
    point_cloud_with_label: {list, np.ndarray} with shape (num_points, 4)
        The point cloud to visualize. The first three columns represent the 
        xyz coordinate of each points, respectively. The last column gives 
        the label of each point. 
    color_dict: dict
        A dictionary records the color for each label.
    scale: tuple
        # TODO.
    '''
    assert type(point_cloud_with_label) == list or \
        type(point_cloud_with_label) == np.ndarray
    if type(point_cloud_with_label) == np.ndarray:
        assert len(point_cloud_with_label.shape) == 2
        assert point_cloud_with_label.shape[1] == 4
        point_cloud_with_label = point_cloud_with_label.tolist()
    else:
        # TODO: check format
        pass

    num_labels = len(color_dict)
    label_list = list(color_dict.keys())

    ax = plt.figure().add_subplot(111, projection='3d')

    for idx in range(num_labels):
        cur_label = label_list[idx]
        cur_point_list = []
        for point in point_cloud_with_label:
            if int(point[3]) == cur_label:
                cur_point_list.append([point[0], point[1], point[2]])
        
        cur_point_list = np.array(cur_point_list)
        ax.scatter(
            cur_point_list[:, 0], 
            cur_point_list[:, 1], 
            cur_point_list[:, 2], 
            c=color_dict[cur_label]
        )
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


if __name__ == '__main__':
    color_dict = {
        PlaneLabel.horizontal.label: PlaneLabel.horizontal.color,
        PlaneLabel.vertical.label: PlaneLabel.vertical.color
    }

    stair1 = create_stairs(num_steps=7, length=0.128, width=1.2, height=0.125, label=True)

    visualize_pc_no_color(stair1[:, 0:3])
    visualize_pc_color(stair1, color_dict)