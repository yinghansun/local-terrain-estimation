import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '../visualization'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../data'))

import numpy as np

from eval_utils import load_model, preprocess_data, eval_pointnet2_model, eval_3dgcn_model
from hyper_paras import *
from label import PlaneLabel
from simple_visualizer import visualize_pc_color, visualize_pc_no_color



cur_path = os.path.dirname(os.path.abspath(__file__))
model_root_path = cur_path + '/../model/'
# model_name = 'para_dic1021.pth'
model_name = 'gcn3d_1124_0.pth'
model_path = model_root_path + model_name
# model = load_model(model_path, NUM_CLASSES, DEVICE, model='pointnet++')
model = load_model(model_path, NUM_CLASSES, DEVICE, model='3dgcn')

# test_data = np.load(cur_path + '/../data/real_data/data0.npy')
test_data = np.load(cur_path + '/../data/real_data/data_1.5m_1.npy')
test = np.zeros((test_data.shape[0], test_data.shape[1]))
test[:, 2] = -test_data[:, 0]
test[:, 1] = test_data[:, 2]
test[:, 0] = test_data[:, 1]
# visualize_pc_no_color(test)
# print(test_data.shape)
# visualize_pc_no_color(test_data)

theta = (5 / 18) * 3.1415926
R = np.array([
    [1, 0, 0],
    [0, np.cos(theta), -np.sin(theta)],
    [0, np.sin(theta), np.cos(theta)]
])
test = (R @ test.T).T

test_data = test

current_points, _ = preprocess_data(test_data[:, 0:3])

# result_pcl = eval_pointnet2_model(current_points, model, DEVICE)
result_pcl = eval_3dgcn_model(current_points, model, DEVICE)

color_dict = {
    PlaneLabel.horizontal.label: PlaneLabel.horizontal.color,
    PlaneLabel.vertical.label: PlaneLabel.vertical.color,
    # PlaneLabel.others.label: PlaneLabel.others.color
}
visualize_pc_color(result_pcl, color_dict)


# print(color_dict[:, 3] == PlaneLabel.horizontal.label)
# print(np.any)

# start_time = time.time()
# for i in range(200):
#     # current_points, _ = preprocess_data(test_data[:, 0:3])
#     result_pcl = eval_3dgcn_model(current_points, model, DEVICE)
# end_time = time.time()
# print(end_time - start_time)