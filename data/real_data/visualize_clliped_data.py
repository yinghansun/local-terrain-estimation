import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../visualization'))

import numpy as np

from simple_visualizer import visualize_pc_no_color

data_path = os.path.join(os.path.dirname(__file__), '2/camera_coordinates_cliped.npy')
test_data = np.load(data_path)

test = np.zeros((test_data.shape[0], test_data.shape[1]))
test[:, 2] = -test_data[:, 0]
test[:, 1] = test_data[:, 2]
test[:, 0] = test_data[:, 1]

theta = -(5 / 18) * 3.1415926
R = np.array([
    [1, 0, 0],
    [0, np.cos(theta), -np.sin(theta)],
    [0, np.sin(theta), np.cos(theta)]
])
test = (R @ test.T).T

test_data = test

visualize_pc_no_color(test_data)