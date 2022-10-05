import os
from typing import List, Optional

import numpy as np
from torch.utils.data import Dataset


class PointcloudDataset(Dataset):

    def __init__(
        self, 
        root_path: str, 
        num_classes: int,
        usage: str,
        num_points: Optional[int] = 2048,
        block_size: Optional[float] = 1.0,
        sample_rate: Optional[float] = 0.5
    ) -> None:
        super().__init__()

        assert usage in root_path

        self.__num_points = num_points
        self.__block_size = block_size

        data_list: List[np.ndarray] = []
        for file_name in list(os.walk(root_path))[0][2]:
            data_list.append(np.load(root_path + file_name))

        self.__points_list: List[np.ndarray] = []
        self.__labels_list: List[np.ndarray] = []
        self.__coor_min, self.__coor_max = [], []
        points_counter_list = []
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
        # print(self.labelweights)

        sample_prob = points_counter_list / np.sum(points_counter_list)
        num_iter = int(np.sum(points_counter_list) * sample_rate / num_points)
        subpointcloud_idxs = []
        for index in range(len(data_list)):
            subpointcloud_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
        # print(subpointcloud_idxs)
        self.__subpointcloud_idxs = np.array(subpointcloud_idxs)
        print("Totally {} samples in training set.".format(len(self.__subpointcloud_idxs)))


    def __getitem__(self, idx):
        data_idx = self.__subpointcloud_idxs[idx]
        points = self.__points_list[data_idx]
        labels = self.__labels_list[data_idx]
        num_points = points.shape[0]

        while True:
            center = points[np.random.choice(num_points)][:3]
            block_min = center - [self.__block_size / 2., self.__block_size / 2., 0.]
            block_max = center + [self.__block_size / 2., self.__block_size / 2., 0.]
            point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
            if point_idxs.size > 512:
                break

        if point_idxs.size >= self.__num_points:
            selected_point_idxs = np.random.choice(point_idxs, self.__num_points, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.__num_points, replace=True)
        
        # normalize
        selected_points = points[selected_point_idxs, :]  # num_point * 3
        current_points = np.zeros((self.__num_points, 9))  # num_point * 9
        current_points[:, 6] = selected_points[:, 0] / self.__coor_max[data_idx][0]
        current_points[:, 7] = selected_points[:, 1] / self.__coor_max[data_idx][1]
        current_points[:, 8] = selected_points[:, 2] / self.__coor_max[data_idx][2]
        current_points[:, 3] = np.zeros((self.__num_points, ))
        current_points[:, 4] = np.zeros((self.__num_points, ))
        current_points[:, 5] = np.zeros((self.__num_points, ))
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        current_points[:, 0:3] = selected_points
        current_labels = labels[selected_point_idxs]

        return current_points, current_labels

    def __len__(self):
        return len(self.__subpointcloud_idxs)

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
    # train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=10, pin_memory=True)
    for idx in range(4):
        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i+1, len(train_loader), time.time() - end))
            end = time.time()


if __name__ == '__main__':
    test()