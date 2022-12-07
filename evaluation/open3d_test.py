import os
import time

import numpy as np
import open3d as o3d

from eval_utils import load_model, eval_3dgcn_model
from hyper_paras import *

cur_path = os.path.dirname(os.path.abspath(__file__))

# test_data = np.load(cur_path + '/../data/basic_shapes/dataset/test/stair9.npy')
# test_data = test_data[:, 0:3]

test_data = np.load(cur_path + '/../data/real_data/data_1.5m_1.npy')
test = np.zeros((test_data.shape[0], test_data.shape[1]))
test[:, 2] = -test_data[:, 0]
test[:, 1] = test_data[:, 2]
test[:, 0] = test_data[:, 1]
theta = (5 / 18) * 3.1415926
R = np.array([
    [1, 0, 0],
    [0, np.cos(theta), -np.sin(theta)],
    [0, np.sin(theta), np.cos(theta)]
])
test = (R @ test.T).T
test_data = test

model_root_path = cur_path + '/../model/'
# model_name = 'para_dic1021.pth'
model_name = 'gcn3d_1124_0.pth'
model_path = model_root_path + model_name
# model = load_model(model_path, NUM_CLASSES, DEVICE, model='pointnet++')
model = load_model(model_path, NUM_CLASSES, DEVICE, model='3dgcn')

def downsample_inference():
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(test_data)
    
    downpcd = pcd.voxel_down_sample(voxel_size=0.01)
    down_points = np.asarray(downpcd.points)

    num_points = down_points.shape[0]

    result_pcl = eval_3dgcn_model(down_points, model, DEVICE, num_points=num_points)

    # np.savetxt('real_1.5m_1.txt', result_pcl)

    down_colors = np.zeros((num_points, 3), dtype=np.float64)
    for pt_idx in range(num_points):
        if result_pcl[pt_idx, 3] == 1.:
            down_colors[pt_idx, 2] = 1
        else:
            down_colors[pt_idx, 0] = 1 
    downpcd.colors = o3d.utility.Vector3dVector(down_colors)

    return downpcd

    # o3d.visualization.draw_geometries([downpcd])

def build_octree(downpcd):
    octree = o3d.geometry.Octree(max_depth=6)
    octree.convert_from_point_cloud(downpcd)
    o3d.visualization.draw_geometries([octree])

if __name__ == '__main__':
    downpcd = downsample_inference()
    # start_time = time.time()
    # for _ in range(500):
    #     build_octree(downpcd)
    # end_time = time.time()
    # print(end_time - start_time)
    build_octree(downpcd)
