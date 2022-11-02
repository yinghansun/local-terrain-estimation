import math
import os
import sys
from typing import List, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), '../../visualization'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import numpy as np
import scipy.linalg as linalg

from label import PlaneLabel
from simple_visualizer import visualize_pc_no_color


def create_horizontal_plane(
    x_upper: float,
    x_lower: float,
    y_upper: float,
    y_lower: float,
    z: float,
    scale: Optional[float] = 0.01,
    label: Optional[bool] = False,
    visualize: Optional[bool] = False
) -> np.ndarray:
    num_points_x = int((x_upper - x_lower) / scale)
    num_points_y = int((y_upper - y_lower) / scale)

    x_list = np.linspace(x_upper, x_lower, num_points_x)
    y_list = np.linspace(y_upper, y_lower, num_points_y)

    if label:
        point_list = np.zeros((num_points_x * num_points_y, 4))
    else:
        point_list = np.zeros((num_points_x * num_points_y, 3))
    idx = 0
    for x in x_list:
        for y in y_list:
            point_list[idx, 0:3] = np.array([x, y, z])
            if label:
                point_list[idx, 3] = PlaneLabel.horizontal.label
            idx += 1

    if visualize:
        scale = (num_points_x, num_points_y, (num_points_x + num_points_y) / 2)
        visualize_pc_no_color(point_list[:, 0:3], scale=scale)

    return point_list


def create_vertical_plane_xfixed(
    y_upper: float,
    y_lower: float,
    z_upper: float,
    z_lower: float,
    x: float,
    scale: Optional[float] = 0.01,
    label: Optional[bool] = False,
    visualize: Optional[bool] = False
) -> np.ndarray:
    num_points_y = int((y_upper - y_lower) / scale)
    num_points_z = int((z_upper - z_lower) / scale)

    y_list = np.linspace(y_upper, y_lower, num_points_y)
    z_list = np.linspace(z_upper, z_lower, num_points_z)

    if label:
        point_list = np.zeros((num_points_y * num_points_z, 4))
    else:
        point_list = np.zeros((num_points_y * num_points_z, 3))
    idx = 0
    for y in y_list:
        for z in z_list:
            point_list[idx, 0:3] = np.array([x, y, z])
            if label:
                point_list[idx, 3] = PlaneLabel.vertical.label
            idx += 1

    if visualize:
        scale = ((num_points_y + num_points_z) / 2, num_points_y, num_points_z)
        visualize_pc_no_color(point_list[:, 0:3], scale=scale)

    return point_list


def create_vertical_plane_yfixed(
    x_upper: float,
    x_lower: float,
    z_upper: float,
    z_lower: float,
    y: float,
    scale: Optional[float] = 0.03,
    label: Optional[bool] = False,
    visualize: Optional[bool] = False
) -> np.ndarray:
    num_points_x = int((x_upper - x_lower) / scale)
    num_points_z = int((z_upper - z_lower) / scale)

    x_list = np.linspace(x_upper, x_lower, num_points_x)
    z_list = np.linspace(z_upper, z_lower, num_points_z)

    if label:
        point_list = np.zeros((num_points_x * num_points_z, 4))
    else:
        point_list = np.zeros((num_points_x * num_points_z, 3))
    idx = 0
    for x in x_list:
        for z in z_list:
            point_list[idx, 0:3] = np.array([x, y, z])
            if label:
                point_list[idx, 3] = PlaneLabel.vertical.label
            idx += 1

    if visualize:
        scale = (num_points_x, (num_points_x + num_points_z) / 2, num_points_z)
        visualize_pc_no_color(point_list[:, 0:3], scale=scale)

    return point_list


def create_sloping_plane_xfixed(
    x_upper: float,
    x_lower: float,
    y_upper: float,
    y_lower: float,
    z_upper: float,
    z_lower: float,
    scale: Optional[float] = 0.03,
    label: Optional[bool] = False,
    visualize: Optional[bool] = False
) -> np.ndarray:
    """Create a sloping plane about x-axis by given two vertex:
    (x_upper, y_upper, z_upper), (x_lower, y_lower, z_lower)
    """

    sin = abs(z_upper - z_lower)/math.sqrt((y_upper - y_lower) ** 2 + (z_upper-z_lower) ** 2)
    num_points_x = abs(int((x_upper - x_lower) / scale))
    num_points_yz = int((z_upper-z_lower)/(sin * scale))

    x_list = np.linspace(x_upper, x_lower, num_points_x)
    y_list = np.linspace(y_upper, y_lower, num_points_yz)
    z_list = np.linspace(z_upper, z_lower, num_points_yz)

    if label:
        point_list = np.zeros((num_points_x * num_points_yz, 4))
    else:
        point_list = np.zeros((num_points_x * num_points_yz, 3))
    idx = 0

    for x in x_list:
        id_z = 0
        for y in y_list:
            z = z_list[id_z]
            id_z += 1
            point_list[idx, 0:3] = np.array([x, y, z])
            if label:
                point_list[idx, 3] = PlaneLabel.sloping.label
            idx += 1

    if visualize:
        scale = (num_points_x, num_points_yz, (num_points_x + num_points_yz) / 2)
        visualize_pc_no_color(point_list[:, 0:3], scale=scale)

    return point_list


def create_sloping_plane_yfixed(
    x_upper: float,
    x_lower: float,
    y_upper: float,
    y_lower: float,
    z_upper: float,
    z_lower: float,
    scale: Optional[float] = 0.03,
    label: Optional[bool] = False,
    visualize: Optional[bool] = False
) -> np.ndarray:
    """Create a sloping plane about y-axis by given two vertex:
        (x_upper, y_upper, z_upper),(x_lower, y_lower, z_lower)
    """

    sin = abs(z_upper - z_lower)/math.sqrt((x_upper - x_lower) ** 2 + (z_upper-z_lower) ** 2)
    num_points_y = abs(int((y_upper - y_lower) / scale))
    num_points_xz = int((z_upper-z_lower)/(sin * scale))

    x_list = np.linspace(x_upper, x_lower, num_points_xz)
    y_list = np.linspace(y_upper, y_lower, num_points_y)
    z_list = np.linspace(z_upper, z_lower, num_points_xz)

    if label:
        point_list = np.zeros((num_points_y * num_points_xz, 4))
    else:
        point_list = np.zeros((num_points_y * num_points_xz, 3))
    idx = 0

    for y in y_list:
        id_z = 0
        for x in x_list:
            z = z_list[id_z]
            id_z += 1
            point_list[idx, 0:3] = np.array([x, y, z])
            if label:
                point_list[idx, 3] = PlaneLabel.sloping.label
            idx += 1

    if visualize:
        scale = (num_points_y, num_points_xz, (num_points_y + num_points_xz) / 2)
        visualize_pc_no_color(point_list[:, 0:3], scale=scale)

    return point_list


def create_sphere(
    r: float,
    centre_x: float,
    centre_y: float,
    centre_z: float,
    label: Optional[bool] = False,
    visualize: Optional[bool] = False
):
    """Create a sphere (base on polar coordinates)
    Paras:
    - centre_x, centre_y, centre_z: the center of sphere
    """
    num_points_theta = 60
    num_points_beta = 30

    theta_list = np.linspace(-np.pi, np.pi, num_points_theta)
    beta_list = np.linspace(-np.pi/2, np.pi/2, num_points_beta)

    if label:
        point_list = np.zeros((num_points_theta * num_points_beta, 4))
    else:
        point_list = np.zeros((num_points_theta * num_points_beta, 3))

    idx = 0
    for beta in beta_list:
        for theta in theta_list:
            point_list[idx, 0:3] = np.array([
                centre_x + r * np.sin(theta) * np.cos(beta),
                centre_y + r * np.sin(theta) * np.sin(beta),
                centre_z + r * np.cos(theta)
                ])
            if label:
                point_list[idx, 3] = PlaneLabel.others.label
            idx += 1

    if visualize:
        scale = (num_points_theta, num_points_beta, (num_points_theta + num_points_beta) / 2)
        visualize_pc_no_color(point_list[:, 0:3], scale=scale)

    return point_list



def create_box(
    length: float,
    width: float,
    height: float,
    center: Optional[np.ndarray] = np.zeros(3),
    label: Optional[bool] = False,
    visualize: Optional[bool] = False
) -> np.ndarray:
    z_top = center[2] + height / 2
    z_bottom = center[2] - height / 2
    y_right = center[1] + width / 2
    y_left = center[1] - width / 2
    x_front = center[0] + length / 2
    x_rear = center[0] - length / 2

    top_point_list = create_horizontal_plane(x_front, x_rear, y_right, y_left, z_top, label=label)
    bottom_point_list = create_horizontal_plane(x_front, x_rear, y_right, y_left, z_bottom, label=label)
    right_point_list = create_vertical_plane_yfixed(x_front, x_rear, z_top, z_bottom, y_right, label=label)
    left_point_list = create_vertical_plane_yfixed(x_front, x_rear, z_top, z_bottom, y_left, label=label)
    front_point_list = create_vertical_plane_xfixed(y_right, y_left, z_top, z_bottom, x_front, label=label)
    rear_point_list = create_vertical_plane_xfixed(y_right, y_left, z_top, z_bottom, x_rear, label=label)

    point_list_list = [top_point_list, bottom_point_list, right_point_list, left_point_list, front_point_list,
                       rear_point_list]
    total_num_points = 0
    for list in point_list_list:
        total_num_points += list.shape[0]
    if label:
        point_list = np.zeros((total_num_points, 4))
    else:
        point_list = np.zeros((total_num_points, 3))
    pointer = 0
    for list in point_list_list:
        point_list[pointer:pointer + list.shape[0], :] = list
        pointer += list.shape[0]

    if visualize:
        visualize_pc_no_color(point_list[:, 0:3], scale=(1, 1, 1))

    return point_list


def create_stair(
    num_steps: int,
    step_length: float,
    width: float,
    step_height: float,
    init_height: Optional[float] = 0.,
    label: Optional[bool] = False,
    visualize: Optional[bool] = False
) -> np.ndarray:
    '''Create a stair by given parameters.

    Args
    ----
    num_steps: int
        number of steps of the stair
    step_length: float
        length per step.
    width: float
        width of the stair.
    step_height: float
        height per step.
    init_height: (optional) float
        initial height of the stair.
    label: (optional) bool
        whether the point cloud has label.
    visualize: (optional) bool
        whether to visualize the boxes or not.

    Returns
    -------
    A stair point cloud (simulated ground truth point cloud).
    '''
    cur_height = init_height
    cur_length = 0
    point_list_list: List[np.ndarray] = []
    total_num_points = 0

    for _ in range(num_steps):
        cur_vertical_pointlist = create_vertical_plane_xfixed(width / 2, -width / 2, cur_height + step_height, cur_height,
                                                              cur_length, label=label)
        point_list_list.append(cur_vertical_pointlist)
        total_num_points += cur_vertical_pointlist.shape[0]
        cur_height += step_height

        cur_horizontal_pointlist = create_horizontal_plane(cur_length + step_length, cur_length, width / 2, -width / 2,
                                                           cur_height, label=label)
        point_list_list.append(cur_horizontal_pointlist)
        total_num_points += cur_horizontal_pointlist.shape[0]
        cur_length += step_length

    if label:
        point_list = np.zeros((total_num_points, 4))
    else:
        point_list = np.zeros((total_num_points, 3))
    pointer = 0
    for list in point_list_list:
        point_list[pointer:pointer + list.shape[0], :] = list
        pointer += list.shape[0]

    if visualize:
        visualize_pc_no_color(point_list[:, 0:3], scale=(1, 1, 1))

    return point_list


def add_noise_pointlist(
    point_list: np.ndarray,
    std: float,
    visualize: Optional[bool] = False
) -> np.ndarray:
    num_data = point_list.shape[0]
    noise = np.random.normal(0, std, num_data * 3)

    for i in range(num_data):
        point_list[i, 0] += noise[3 * i]
        point_list[i, 1] += noise[3 * i + 1]
        point_list[i, 2] += noise[3 * i + 2]

    if visualize:
        visualize_pc_no_color(point_list[:, 0:3], scale=(1, 1, 1))

    return point_list


def rotate_pointlist(
    point_list: np.ndarray,
    theta: float,
    axis_x: Optional[int] = 0,
    axis_y: Optional[int] = 0,
    axis_z: Optional[int] = 1
) -> np.ndarray:
    """Rotate a point list about z-axis by a given angle theta
    Paras:
    - point_list: point list to be rotated
    - theta: rotation angle about z-axis (in radius)
    """

    rand_axis = [axis_x, axis_y, axis_z]
    rot_matrix = rotate_mat(rand_axis, theta)

    for point in point_list:
        coor = np.array([point[0], point[1], point[2]])
        coor_trans = rot_matrix @ coor
        point[0:3] = coor_trans

    return point_list


def rotate_mat(axis, theta):
    """
    create a specified rotation matrix
    Paras:
    - axis: axis of rotation (through the (0, 0, 0))
    - theta: rotation angle about axis (in radius)
    """
    rot_matrix = linalg.expm(np.cross(np.eye(3), axis / np.linalg.norm(axis) * theta))
    return rot_matrix


def translate_pointlist(
        point_list: np.ndarray,
        xt: float,
        yt: float,
        zt: float
):
    """Translate a point list
    Paras:
    - point_list: point list to be rotated
    - xt, yt, zt: the distance of moving over x-axis, y-axis, z-axis
    """
    tranlate_mat = np.array([
        [1, 0, 0, xt],
        [0, 1, 0, yt],
        [0, 0, 1, zt],
        [0, 0, 0, 1]
    ])

    for point in point_list:
        coor = np.array([point[0], point[1], point[2], 1])
        coor_trans = tranlate_mat @ coor
        point[0:3] = coor_trans[0:3]

    return point_list


if __name__ == '__main__':
    # point_list = create_horizontal_plane(0.5, -0.5, -0.2, -0.4, 0.3, label=True, visualize=True)
    # point_list = create_vertical_plane_xfixed(0.5, -0.5, -0.2, -0.4, 0.3, label=True, visualize=True)
    # point_list = create_vertical_plane_yfixed(0.5, -0.5, -0.2, -0.4, 0.3, label=True, visualize=True)
    # point_list = create_sloping_plane_xfixed(0.5, -0.5, -0.2, -0.4, 0.3, -0.3, label=True, visualize=True)
    # point_list = create_sloping_plane_yfixed(0.5, -0.5, -0.2, -0.4, 0.3, -0.3, label=True, visualize=True)
    # point_list = create_sphere(0.5, 0.5, 0.5, 0.5, label=True, visualize=True)

    # point_list = create_box(1, 1, 0.5, label=True, visualize=True)
    point_list = create_stair(4, 0.3, 1, 0.25, init_height=0.2, label=True, visualize=True)
    print(point_list.shape)
    # point_list = add_noise_pointlist(point_list, 0.01)
    # point_list = rotate_pointlist(point_list, np.pi / 6, axis_x=1, axis_y=0, axis_z=0)
    # point_list = translate_pointlist(point_list, 1, 1, 1)

    visualize_pc_no_color(point_list[:, 0:3])