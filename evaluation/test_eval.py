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
model_name = 'para_dic1021.pth'
model_path = model_root_path + model_name
model = load_model(model_path, NUM_CLASSES, DEVICE)

test_data = np.load(cur_path + '/../data/basic_shapes/dataset/test/stair9.npy')

current_points, _ = preprocess_data(test_data[:, 0:3])

result_pcl = eval_pointnet2_model(current_points, model, DEVICE)

color_dict = {
    PlaneLabel.horizontal.label: PlaneLabel.horizontal.color,
    PlaneLabel.vertical.label: PlaneLabel.vertical.color,
    # PlaneLabel.others.label: PlaneLabel.others.color
}
visualize_pc_color(result_pcl, color_dict)


num_points = NUM_POINTS
horizontal_points = []
vertical_points = []
for idx in range(num_points):
    if result_pcl[idx, 3] == PlaneLabel.horizontal.label:
        horizontal_points.append(result_pcl[idx, :])
    elif result_pcl[idx, 3] == PlaneLabel.vertical.label:
        vertical_points.append(result_pcl[idx, :])

horizontal_points = np.array(horizontal_points)
vertical_points = np.array(vertical_points)

# visualize_pc_no_color(horizontal_points[:, 0:3])
# visualize_pc_no_color(vertical_points[:, 0:3])

horizontal_sorted = horizontal_points[np.argsort(horizontal_points[:, 2])]
print(horizontal_sorted)
z_list = []
# for i in range(horizontal_sorted):
    