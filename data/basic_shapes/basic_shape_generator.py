import os

import numpy as np
from basic_shapes_utils import *

cur_path = os.path.dirname(os.path.abspath(__file__))
data_root_path = cur_path + '/dataset/'
# train_path = data_root_path + '/train/'
# test_path = data_root_path + '/test/'

if not os.path.exists(data_root_path):
    os.mkdir(data_root_path)
# if not os.path.exists(train_path):
#     os.mkdir(train_path)
# if not os.path.exists(test_path):
#     os.mkdir(test_path)

stair1 = create_stairs(num_steps=7, length=0.128, width=1.2, height=0.125, label=True)
np.save(data_root_path + 'stair1.npy', stair1)

stair2 = create_stairs(num_steps=5, length=0.225, width=1.3, height=0.2, label=True)
np.save(data_root_path + 'stair2.npy', stair2)


for i in range(10):
    name = 'box' + str(i) + '.npy'
    length = np.random.rand()
    width = np.random.rand()
    height = np.random.rand()

    if np.random.rand() > 0.5:
        length += 1
    else:
        length += 1.5

    if np.random.rand() > 0.5:
        width += 0.365
    else:
        width += 0.983

    if np.random.rand() > 0.5:
        height += 0.454
    else:
        height += 1.344

    box_i = create_box(length=length, width=width, height=height, label=True)
    angle = np.random.rand() * 3.1415926
    box_i = rotate_pointlist(box_i, angle)
    box_i = add_noise_pointlist(box_i, 0.005)
    # visualize_pointlist(box_i)
    np.save(data_root_path + name, box_i)