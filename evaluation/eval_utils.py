import os
import sys
from typing import Optional, Union

sys.path.append(os.path.join(os.path.dirname(__file__), '../3rd_party/pointnet_pointnet2_pytorch/'))

from dgl.geometry import farthest_point_sampler
import numpy as np
import torch

from hyper_paras import *
from pointnet2_sem_seg import PointNet2


def downsample_pcl(
    points: Union[torch.Tensor, np.ndarray], 
    num_ds: int, 
):
    '''Down sample the point cloud using FPS (Farthest Point Sampling).

    Args
    ----
    points: torch.Tensor or np.ndarray
        point cloud to be downsampled.
    num_ds: int
        target point number after downsampling.
    '''
    is_numpy = isinstance(points, np.ndarray)
    is_tensor = isinstance(points, torch.Tensor)
    assert is_numpy or is_tensor

    if is_numpy:
        points = torch.from_numpy(points)

    ids = farthest_point_sampler(points, num_ds)
    ids_reshape = ids.unsqueeze(-1).repeat(1, 1, 3)
    out_pcl = torch.gather(points, 1, ids_reshape)
    if is_numpy:
        out_pcl = out_pcl.numpy()
    return out_pcl, ids


def preprocess_data(
    points: np.ndarray,
    num_points_des: Optional[int] = NUM_POINTS
):
    '''Preprocess the data for training and testing.

    Args
    ----
    points: np.ndarray
        point cloud to be preprocessed.
    num_points_des: (optional) int
        identify the target point after downsampling.
    '''
    coor_max = np.amax(points, axis=0)[:3]

    points = np.expand_dims(points, 0)
    selected_points, ids = downsample_pcl(points, num_points_des)
    selected_points = selected_points.squeeze(axis=0)
    center = np.mean(selected_points, axis=0)
    # normalize
    current_points = np.zeros((num_points_des, 6))
    current_points[:, 3] = selected_points[:, 0] / coor_max[0]
    current_points[:, 4] = selected_points[:, 1] / coor_max[1]
    current_points[:, 5] = selected_points[:, 2] / coor_max[2]
    selected_points[:, 0] = selected_points[:, 0] - center[0]
    selected_points[:, 1] = selected_points[:, 1] - center[1]
    current_points[:, 0:3] = selected_points

    return current_points, ids


def load_model(
    model_path: str,
    num_classes: int,
    device: str,
):
    '''Load trained pytorch model for testing.

    Args
    ----
    model_path: str
        path of the model.
    num_classes: int
        total number of classes for the data.
    device: str, {'cuda', 'cpu'}
        identify the device for torch.
    '''
    state_dict = torch.load(model_path)
    model = PointNet2(num_classes).to(device)
    model.load_state_dict(state_dict)
    return model


def eval_pointnet2_model(
    points: np.ndarray,
    model,
    device: str,
    num_points: Optional[int] = NUM_POINTS
):
    '''Evaluate given point cloud using given trained pointnet++ model.

    Args
    ----
    '''
    points = torch.from_numpy(points).unsqueeze(0).float().to(device)
    # points = points.float().to(device)
    points = points.transpose(2, 1)

    seg_pred, _ = model(points)
    pred_val = seg_pred.contiguous().cpu().data.numpy()
    pred_val = np.argmax(pred_val, 2)
    points = points.cpu().numpy()[0][0:3, :].T

    result_pcl = np.zeros((num_points, 4))
    result_pcl[:, 0:3] = points
    result_pcl[:, 3] = pred_val

    return result_pcl