import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../visualization'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../data'))

import numpy as np

from eval_utils import load_model, preprocess_data, eval_pointnet2_model
from hyper_paras import *
from label import PlaneLabel
from simple_visualizer import visualize_pc_color, visualize_pc_no_color


cur_path = os.path.dirname(os.path.abspath(__file__))
model_root_path = cur_path + '/../model/'
model_name = 'para_dic1010.pth'
model_path = model_root_path + model_name
model = load_model(model_path, NUM_CLASSES, DEVICE)

test_data = np.load(cur_path + '/../data/real_data/data0.npy')
test = np.zeros((test_data.shape[0], test_data.shape[1]))
test[:, 2] = -test_data[:, 0]
test[:, 1] = test_data[:, 2]
test[:, 0] = test_data[:, 1]
visualize_pc_no_color(test)
# print(test_data.shape)
# visualize_pc_no_color(test_data)

test_data = test

current_points, _ = preprocess_data(test_data[:, 0:3])

result_pcl = eval_pointnet2_model(current_points, model, DEVICE)

color_dict = {
    PlaneLabel.horizontal.label: PlaneLabel.horizontal.color,
    PlaneLabel.vertical.label: PlaneLabel.vertical.color,
    PlaneLabel.others.label: PlaneLabel.others.color
}
visualize_pc_color(result_pcl, color_dict)
