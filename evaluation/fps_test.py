import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../data/basic_shapes'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../visualization'))

from dgl.geometry import farthest_point_sampler
import numpy as np
import torch

from basic_shapes_utils import create_box
from simple_visualizer import visualize_pc_no_color


def downsample_pcl(batch_pcl, num):
    '''Down sample the point cloud using farthest point sampling

    Paras
    -----
    batch_pcl: torch.tensor or np.ndarray
        batch x number
    num: int
        target point number
    '''
    is_numpy = isinstance(batch_pcl, np.ndarray)
    if is_numpy:
        batch_pcl = torch.from_numpy(batch_pcl)

    ids = farthest_point_sampler(batch_pcl, num)
    print(ids)
    ids = ids.unsqueeze(-1).repeat(1, 1, 3)
    print(ids)
    out = torch.gather(batch_pcl, 1, ids)
    if is_numpy:
        out = out.numpy()
    return out


if __name__ == '__main__':
    pcl = create_box(0.8, 0.8, 0.5)
    visualize_pc_no_color(pcl)
    pcl = np.expand_dims(pcl, 0)
    pcl_downsample = downsample_pcl(pcl, 1024)
    pcl_downsample = pcl_downsample.squeeze(axis=0)
    visualize_pc_no_color(pcl_downsample)
