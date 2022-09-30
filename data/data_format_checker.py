import numpy as np

def check_pointcloud_format(
    pointcloud: np.ndarray
) -> bool:
    '''Check whether the data in standard form.

    Paras
    -----
    pointcloud: np.ndarray
        data to be checked.
    
    Returns
    -------
    has_label: bool
        whether the data includes labels for each point.
    '''
    assert type(pointcloud) == np.ndarray, \
        'The data type of pointcloud must be numpy array.'
    assert len(pointcloud.shape) == 2, \
        'The dimension of array must be 2.'

    if pointcloud.shape[1] == 3:
        has_label = False
    elif pointcloud.shape[1] == 4:
        has_label = True
    else:
        raise ValueError('Number of columns of the data array must be 3 (no label) or 4 (with label).')

    return has_label

if __name__ == '__main__':
    data1 = np.array([
        [1, 2, 3],
        [2, 4, 6]
    ])
    has_label1 = check_pointcloud_format(data1)
    print(has_label1)

    data2 = np.array([
        [1, 2, 3, 0],
        [2, 4, 6, 1]
    ])
    has_label2 = check_pointcloud_format(data2)
    print(has_label2)

    data3 = np.array([
        [1, 2],
        [2, 4]
    ])
    has_label3 = check_pointcloud_format(data3)
    print(has_label3)