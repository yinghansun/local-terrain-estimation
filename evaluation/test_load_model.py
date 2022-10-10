import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../3rd_party/pointnet_pointnet2_pytorch/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../visualization'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../data'))


import numpy as np
import torch

from BasicShapeDataloaderInterface import PointcloudDataset
from pointnet2_sem_seg import PointNet2
from label import PlaneLabel
from simple_visualizer import visualize_pc_color



NUM_CLASSES = 3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


cur_path = os.path.dirname(os.path.abspath(__file__))
model_path = cur_path + '/../model/'
# model_name = 'test_para_dict.pth'
model_name = 'para_dic1010.pth'

state_dict = torch.load(model_path + model_name)
# print(state_dict)
classifier = PointNet2(NUM_CLASSES).to(DEVICE)
classifier.load_state_dict(state_dict)
classifier.eval()

# test_data = np.load(cur_path + '/../data/basic_shapes/dataset/test/box0.npy').astype(np.float32)
# # print(test_data)

# num_points = test_data.shape[0]
# coor_min = np.amin(test_data, axis=0)[:3]
# coor_max = np.amax(test_data, axis=0)[:3]
# print(coor_min)

# test_data_ = np.zeros((num_points, 9), dtype=np.float32)
# test_data_[:, 6] = test_data[:, 0] / coor_max[0]
# test_data_[:, 7] = test_data[:, 1] / coor_max[1]
# test_data_[:, 8] = test_data[:, 2] / coor_max[2]
# test_data_[:, 3] = np.zeros((num_points, ), dtype=np.float32)
# test_data_[:, 4] = np.zeros((num_points, ), dtype=np.float32)
# test_data_[:, 5] = np.zeros((num_points, ), dtype=np.float32)
# test_data_[:, 0] = test_data[:, 0]
# test_data_[:, 1] = test_data[:, 1]
# test_data_[:, 0:3] = test_data[:, 0:3]

# print(test_data_.shape)

# test_data_ = torch.tensor(test_data_).view(1, 9, 21588).float().to(DEVICE)
# print(test_data_.size())
# prediction, trans_feat = classifier(test_data_)
# prediction = prediction.contiguous().view(-1, NUM_CLASSES)
# pred_choice = prediction.cpu().data.max(1)[1].numpy()
# print(pred_choice.shape)

# data_pred = np.zeros((num_points, 4))
# data_pred[:, 0:3] = test_data[:, 0:3]
# data_pred[:, 3] = pred_choice
# print(data_pred.shape)


color_dict = {
    PlaneLabel.horizontal.label: PlaneLabel.horizontal.color,
    PlaneLabel.vertical.label: PlaneLabel.vertical.color,
    PlaneLabel.others.label: PlaneLabel.others.color
}

BATCH_SIZE = 1
NUM_WORKERS = 10

cur_path = os.path.dirname(os.path.abspath(__file__))
train_root_path = cur_path + '/../data/basic_shapes/dataset/train/'
train_set = PointcloudDataset(train_root_path, NUM_CLASSES, 'train')
test_root_path = cur_path + '/../data/basic_shapes/dataset/test/'
test_set = PointcloudDataset(test_root_path, NUM_CLASSES, 'test')

test_data_loader = torch.utils.data.DataLoader(
        test_set, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True, 
        drop_last=True
)

pc = np.zeros((2048, 4))

for i, (points, labels) in enumerate(test_data_loader):
    points = points.float().to(DEVICE)
    labels = labels.long().to(DEVICE)
    points = points.transpose(2, 1)

    seg_pred, trans_feat = classifier(points)
    pred_val = seg_pred.contiguous().cpu().data.numpy()
    seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

    pred_val = np.argmax(pred_val, 2)

    if i < 10:
        print(pred_val)
        points = points.cpu().numpy()[0][0:3, :].T
        print(points.shape)
        print(pred_val[0].shape)

        pc[:, 0:3] = points
        pc[:, 3] = pred_val

        visualize_pc_color(pc, color_dict)
    else:
        break




