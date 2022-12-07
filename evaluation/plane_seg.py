import os
import random
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '../visualization'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../data'))

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

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
# test_data = np.load(cur_path + '/../data/basic_shapes/dataset/test/stair9.npy')

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
# test_data = test_data[:, 0:3]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(test_data)

downpcd = pcd.voxel_down_sample(voxel_size=0.01)
pcd_tree = o3d.geometry.KDTreeFlann(downpcd)
    
# o3d.visualization.draw_geometries([downpcd])
down_points = np.asarray(downpcd.points)


# current_points, _ = preprocess_data(test_data[:, 0:3])

# # result_pcl = eval_pointnet2_model(current_points, model, DEVICE)
result_pcl = eval_3dgcn_model(down_points, model, DEVICE, num_points=down_points.shape[0])

color_dict = {
    PlaneLabel.horizontal.label: PlaneLabel.horizontal.color,
    PlaneLabel.vertical.label: PlaneLabel.vertical.color,
    # PlaneLabel.others.label: PlaneLabel.others.color
}
visualize_pc_color(result_pcl, color_dict)

# start_time = time.time()
# for i in range(200):
    # downpcd = pcd.voxel_down_sample(voxel_size=0.01)
    # down_points = np.asarray(downpcd.points)
    # result_pcl = eval_3dgcn_model(down_points, model, DEVICE, num_points=down_points.shape[0])
    # pcd_tree = o3d.geometry.KDTreeFlann(downpcd)
# end_time = time.time()
# print(end_time - start_time)

class Plane:

    def __init__(self, seed, delta=0.03) -> None:
        self.label = seed[3]
        self.points = [seed[0:3]]
        self.num_points = 1
        self.lower_bound = np.array(seed[0:3])
        self.upper_bound = np.array(seed[0:3])
        self.delta = delta
        # self.normal_vector
        # self.MSE

    def update_boundary(self, new_points):
        if new_points[0] < self.lower_bound[0]:
            self.lower_bound[0] = new_points[0]
        if new_points[1] < self.lower_bound[1]:
            self.lower_bound[1] = new_points[1]
        if new_points[2] < self.lower_bound[2]:
            self.lower_bound[2] = new_points[2]

        if new_points[0] > self.upper_bound[0]:
            self.upper_bound[0] = new_points[0]
        if new_points[1] > self.upper_bound[1]:
            self.upper_bound[1] = new_points[1]
        if new_points[2] > self.upper_bound[2]:
            self.upper_bound[2] = new_points[2]

    def add_point(self, new_points):
        # if self.check_add_condition(new_points):
        self.update_boundary(new_points)
        self.num_points += 1
        self.points.append(new_points[0:3])

    def check_add_condition(self, new_points):
        # check if point exists:
        if new_points == None:
            return False
        # check label
        if new_points[3] != self.label:
            return False
        # check distance
        if new_points[0] > self.lower_bound[0] - self.delta and \
           new_points[1] > self.lower_bound[1] - self.delta and \
           new_points[2] > self.lower_bound[2] - self.delta and \
           new_points[0] < self.upper_bound[0] + self.delta and \
           new_points[1] < self.upper_bound[1] + self.delta and \
           new_points[2] < self.upper_bound[2] + self.delta:
            return True
        else:
            return False

    def visualize_plane(self):
        visualize_pc_no_color(self.points)

def check_seed_condition(seed_idx, num_neighbors=5, distance_threshold=0.03):
    if l_result_pcl[seed_idx] == None:
        return False
    k, idxs, _ = pcd_tree.search_knn_vector_3d(downpcd.points[seed_idx], num_neighbors)
    seed_label = result_pcl[seed_idx, 3]
    seed_coordinate = result_pcl[seed_idx, 0:3]
    # check label:
    for idx in list(idxs):
        cur_neighbor_label = result_pcl[idx, 3]
        if cur_neighbor_label != seed_label:
            return False
    # check distance:
    for idx in list(idxs):
        cur_neighbor_coordinate = result_pcl[idx, 0:3]
        distance = np.sqrt(np.sum((cur_neighbor_coordinate - seed_coordinate) ** 2))
        if distance > distance_threshold:
            return False
    return True

def check_outliers(seed_idx, search_radius=0.02):
    k, idx, _ = pcd_tree.search_radius_vector_3d(downpcd.points[seed_idx], search_radius)
    if len(list(idx)) == 1:
        return True
    else:
        return False
# colors = np.zeros((down_points.shape[0], 3))
# seed_idx = 1000
# idxs = check_seed_condition(seed_idx, 5)
# colors[seed_idx, :] = [1, 0, 0]
# for idx in list(idxs):
#     colors[idx, :] = [0, 1, 0]
# downpcd.colors = o3d.utility.Vector3dVector(colors)
# o3d.visualization.draw_geometries([downpcd])

start_time = time.time()
num_neighbors = 5
plane_list = []
outliers = []
l_result_pcl: list = result_pcl.tolist()
l_idx = list(range(len(l_result_pcl)))
num_points = len(l_result_pcl)
cur_selected_points = 0
while cur_selected_points < num_points / 10 * 9:
# for i in range(10):
    while True:
        # seed_idx = random.randint(0, len(l_result_pcl)-1)
        seed_idx = l_idx[random.randint(0, len(l_idx)-1)]
        if check_outliers(seed_idx):
            l_result_pcl[seed_idx] = None
            l_idx.remove(seed_idx)
            cur_selected_points += 1
            print('{}, outlier'.format(seed_idx))
        # print(list(idx))
        if check_seed_condition(seed_idx):
            break
    # print(seed_idx)
    seed = l_result_pcl[seed_idx]
    cur_plane = Plane(seed)
    plane_list.append(cur_plane)
    l_result_pcl[seed_idx] = None
    l_idx.remove(seed_idx)
    cur_selected_points += 1
    
    _, neighbor_idxs, _ = pcd_tree.search_knn_vector_3d(downpcd.points[seed_idx], num_neighbors)
    l_cur_seed_idxs = list(neighbor_idxs)
    # print(l_cur_seed_idxs)
    l_cur_seed = []
    for idx in l_cur_seed_idxs:
        if l_result_pcl[idx] != None:
            l_cur_seed.append(l_result_pcl[idx])
    
    is_first_run = True
    l_next_seed_idxs = []
    while True:
    # for i in range(10):
        if not is_first_run:
            if l_next_seed_idxs == []:
                break
            else:
                l_cur_seed_idxs = l_next_seed_idxs
                l_next_seed_idxs = []
        else:
            is_first_run = False
        for idx in l_cur_seed_idxs:
            cur_pt = l_result_pcl[idx]
            addable = cur_plane.check_add_condition(cur_pt)
            if addable:
                cur_plane.add_point(cur_pt)
                l_idx.remove(idx)
                l_result_pcl[idx] = None
                cur_selected_points += 1

                _, next_neighbor_idxs, _ = pcd_tree.search_knn_vector_3d(downpcd.points[idx], num_neighbors)
                for next_idx in list(next_neighbor_idxs):
                    if l_result_pcl[next_idx] != None:
                        l_next_seed_idxs.append(next_idx)
        # print(l_next_seed_idxs)
        # print(cur_plane.num_points)
        # print(l_result_pcl)
    # print(l_next_seed_idxs)
    # print(len(l_next_seed_idxs))
    # print(cur_plane.label)
    # print(cur_plane.num_points)
    # print(cur_plane.lower_bound)
    # print(cur_plane.upper_bound)
    # # print(cur_plane.points)
    # cur_plane.visualize_plane()
    print(len(plane_list))
    print(num_points)
    print(cur_selected_points)
    print(len(l_idx))
    # break


    
    # break
    # cur_selected_points += 1
end_time = time.time()
print(end_time - start_time)

# select = []
# num_plane = len(plane_list)
# delta = 1 / num_plane
# for i in range(num_plane):
#     select.append(i * delta)
# select = tuple(select)
colors = ['#3682be','#45a776','#f05326','#eed777','#334f65','#b3974e','#38cb7d','#ddae33','#844bb3','#93c555','#5f6694','#df3881']
ax = plt.figure().add_subplot(111, projection='3d')
idx = 0
for plane in plane_list:
    points = np.array(plane.points)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    idx += 1
# plt.gca().set_box_aspect(scale)
plt.xlabel('x')
plt.ylabel('y')
plt.show()



