import os
from typing import List, Optional

import numpy as np
from torch.utils.data import Dataset

from eval_utils import preprocess_data
from hyper_paras import *


class PointcloudDataset(Dataset):

    def __init__(
        self, 
        root_path: str, 
        num_classes: int,
        usage: str,
        num_points: Optional[int] = NUM_POINTS,
        model: Optional[str] = '3dgcn'
    ) -> None:
        '''A point cloud dataloader for Pytorch.

        Attributes
        ----------
        root_path: str
            root path for dataset.
        num_classes: int
            total number of classes.
        usage: str, {'train', 'test'}
            identify the usage of the data (for training or testing).
        num_points:
            number of points of one data after pre-processing.
        '''
        super().__init__()

        assert usage in root_path

        self.__num_points = num_points
        self.__model = model

        data_list: List[np.ndarray] = []
        for file_name in list(os.walk(root_path))[0][2]:
            data_list.append(np.load(root_path + file_name))

        self.__points_list: List[np.ndarray] = []
        self.__labels_list: List[np.ndarray] = []
        self.__coor_min, self.__coor_max = [], []
        points_counter_list = []  # record number of points for each data
        labelcounter = np.zeros(num_classes)

        for data in data_list:
            points, labels = data[:, 0:3], data[:, 3]    # x, y, z, label
            self.__points_list.append(points)
            self.__labels_list.append(labels)
            points_counter_list.append(data.shape[0])

            # print(data.shape[0])
            tmp, _ = np.histogram(labels, range(0, num_classes+1))
            labelcounter += tmp
            # print(tmp)

            coor_min = np.amin(points, axis=0)[:3]
            coor_max = np.amax(points, axis=0)[:3]
            self.__coor_min.append(coor_min)
            self.__coor_max.append(coor_max)

        labelcounter = labelcounter.astype(np.float32)
        labelweights = labelcounter / np.sum(labelcounter)
        # print(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)

        print("Totally {} samples in {} set.".format(len(self.__points_list), usage))

    def __getitem__(self, idx):
        data_idx = idx
        points = self.__points_list[data_idx]
        labels = self.__labels_list[data_idx]

        current_points, ids = preprocess_data(points, model=self.__model)
        current_labels = labels[ids]

        return current_points, current_labels

    def __len__(self):
        return len(self.__points_list)

    @property
    def num_points(self):
        return self.__num_points


def test():
    cur_path = os.path.dirname(os.path.abspath(__file__))
    data_root_path = cur_path + '/../data/basic_shapes/dataset/train/'
    dataset = PointcloudDataset(data_root_path, 3, 'train')
    print('point data size:', dataset.__len__())
    print('point data 0 shape:', dataset.__getitem__(0)[0].shape)
    print('point label 0 shape:', dataset.__getitem__(0)[1].shape)

    import torch, time, random
    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=10, pin_memory=True, worker_init_fn=worker_init_fn)
    for idx in range(4):
        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i+1, len(train_loader), time.time() - end))
            end = time.time()


if __name__ == '__main__':
    test()