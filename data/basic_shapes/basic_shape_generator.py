import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../visualization'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import numpy as np

from basic_shapes_utils import *
from data_augmentation_utils import *
from simple_visualizer import visualize_pc_no_color


def generate_boxes_only(
    num_boxes: int,
    width_range: Optional[tuple] = (0.2, 2.0),
    length_range: Optional[tuple] = (0.2, 2.0),
    height_range: Optional[tuple] = (0.08, 0.25),
    visualize: Optional[bool] = False,
    save: Optional[bool] = False,
    root_path: Optional[str] = None,
):
    '''Generate dataset only contains single box and noise.

    Args
    ----
    num_boxes: int
        number of boxes contained in this dataset.
    width_range: (optional) tuple
        identify the min and max bound of box's width.
    length_range: (optional) tuple
        identify the min and max bound of box's length.
    height_range: (optional) tuple
        identify the min and max bound of box's height.
    visualize: (optional) bool
        whether to visualize the boxes or not.
    save: (optional) bool
        whether to save the boxes or not.
    root_path: (optional) str
        root path of the dataset.
    '''

    for i in range(num_boxes):
        name = 'box' + str(i) + '.npy'

        length = length_range[0] + \
            (length_range[1] - length_range[0]) * np.random.rand()
        width = width_range[0] + \
            (width_range[1] - width_range[0]) * np.random.rand()
        height = height_range[0] + \
            (height_range[1] - height_range[0]) * np.random.rand()

        box_i = create_box(length=length, width=width, height=height, label=True)
        angle = np.random.rand() * 3.1415926
        box_i = rotate_pointlist(box_i, angle)
        box_i = add_noise_pointlist(box_i, 0.05)
        box_i = remove_random_patches(box_i)
        box_i = add_random_clusters(box_i)

        print('Successfully create box with length = ' + \
            '{}, width = {}, height = {}.'.format(length, width, height))

        if visualize:
            visualize_pc_no_color(box_i[:, 0:3])
        if save:
            np.save(root_path + name, box_i)


def generate_boxes_on_plane(
    num_boxes: int,
    width_range: Optional[tuple] = (0.2, 2.0),
    length_range: Optional[tuple] = (0.2, 2.0),
    height_range: Optional[tuple] = (0.1, 1.5),
    visualize: Optional[bool] = False,
    save: Optional[bool] = False,
    root_path: Optional[str] = None,
):
    '''Generate dataset only contains single box and noise.

    Args
    ----
    num_boxes: int
        number of boxes contained in this dataset.
    width_range: (optional) tuple
        identify the min and max bound of box's width.
    length_range: (optional) tuple
        identify the min and max bound of box's length.
    height_range: (optional) tuple
        identify the min and max bound of box's height.
    visualize: (optional) bool
        whether to visualize the boxes or not.
    save: (optional) bool
        whether to save the boxes or not.
    root_path: (optional) str
        root path of the dataset.
    '''

    for i in range(num_boxes):
        name = 'box_plane' + str(i) + '.npy'

        length = length_range[0] + \
            (length_range[1] - length_range[0]) * np.random.rand()
        width = width_range[0] + \
            (width_range[1] - width_range[0]) * np.random.rand()
        height = height_range[0] + \
            (height_range[1] - height_range[0]) * np.random.rand()

        plane = create_horizontal_plane(1.5, -1.5, 1.5, -1.5, 0, label=True)
        print(plane.shape)
        box_i = create_box(length=length, width=width, height=height, center=np.array([0, 0, height/2]), label=True)
        print(box_i.shape)
        data_i = np.vstack((plane, box_i))
        print(data_i.shape)
        angle = np.random.rand() * 3.1415926
        # data_i = rotate_pointlist(data_i, angle)
        # data_i = add_noise_pointlist(data_i, 0.05)
        # data_i = remove_random_patches(data_i)
        # data_i = add_random_clusters(data_i)

        print('Successfully create box with length = ' + \
            '{}, width = {}, height = {}.'.format(length, width, height))

        if visualize:
            visualize_pc_no_color(data_i[:, 0:3])
        if save:
            np.save(root_path + name, data_i)


def generate_stairs_only(
    num_stairs: int,
    base_height: Optional[float] = 0.,
    num_steps_range: Optional[tuple] = (2, 10),
    step_length_range: Optional[tuple] = (0.15, 0.5),
    width_range: Optional[tuple] = (0.2, 2.0),
    step_height_range: Optional[tuple] = (0.08, 0.25),
    visualize: Optional[bool] = False,
    save: Optional[bool] = False,
    root_path: Optional[str] = None,
):
    '''Generate dataset only contains single box and noise.

    Args
    ----
    num_stairs: int
        number of stairs contained in this dataset.
    base_height: (optional) float
        base height of the stairs.
    num_steps_range: (optional) tuple
        identify the min and max bound of stair's step.
    step_length_range: (optional) tuple
        identify the min and max bound of stair's step length.
    width_range: (optional) tuple
        identify the min and max bound of stair's width.
    step_height_range: (optional) tuple
        identify the min and max bound of stair's step_height.
    visualize: (optional) bool
        whether to visualize the stairs or not.
    save: (optional) bool
        whether to save the stairs or not.
    root_path: (optional) str
        root path of the dataset.
    '''

    for i in range(num_stairs):
        name = 'stair' + str(i) + '.npy'

        step_length = step_length_range[0] + \
            (step_length_range[1] - step_length_range[0]) * np.random.rand()
        width = width_range[0] + \
            (width_range[1] - width_range[0]) * np.random.rand()
        step_height = step_height_range[0] + \
            (step_height_range[1] - step_height_range[0]) * np.random.rand()

        num_steps = num_steps_range[0] + \
            np.random.randint(num_steps_range[1] - num_steps_range[0])

        stair_i = create_stair(num_steps, step_length, width, step_height, base_height, label=True)
        angle = np.random.rand() * 3.1415926
        stair_i = rotate_pointlist(stair_i, angle)
        # stair_i = disturb_position(stair_i)
        # stair_i = disturb_height(stair_i)
        stair_i = add_noise_pointlist(stair_i, 0.02)
        stair_i = remove_random_patches(stair_i)
        stair_i = add_random_clusters(stair_i)

        print('Successfully create stair with width = ' + \
            '{}, step length = {}, step height = {}, '.format(width, step_length, step_height) + \
            'number steps = {}.'.format(num_steps))

        if visualize:
            visualize_pc_no_color(stair_i[:, 0:3])
        if save:
            np.save(root_path + name, stair_i)


if __name__ == '__main__':
    cur_path = os.path.dirname(os.path.abspath(__file__))
    data_root_path = cur_path + '/dataset/'
    train_path = data_root_path + '/train/'
    test_path = data_root_path + '/test/'

    if not os.path.exists(data_root_path):
        os.mkdir(data_root_path)
    if not os.path.exists(train_path):
        os.mkdir(train_path)
    if not os.path.exists(test_path):
        os.mkdir(test_path)

    # generate_boxes_only(10, save=False, visualize=True)
    # generate_boxes_only(100, save=True, root_path=train_path)
    # generate_boxes_only(100, save=True, root_path=test_path)

    generate_boxes_on_plane(10, save=False, visualize=True)

    # generate_stairs_only(10, save=False, visualize=True)
    # generate_stairs_only(100, save=True, root_path=train_path)
    # generate_stairs_only(100, save=True, root_path=test_path)