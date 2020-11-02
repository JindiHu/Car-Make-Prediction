import scipy.io as sio
import numpy as np


def get_data():
    mat = sio.loadmat('./data/cars_annos.mat')
    annotations = mat["annotations"]
    _, total_size = annotations.shape
    dataset = np.zeros((total_size, 6))
    for i in range(total_size):
        img_path = annotations[:, i][0][0][0]
        img_name = img_path.split(".")[0][8:]
        img_id = int(img_name) - 1
        for j in range(6):
            dataset[img_id, j] = int(annotations[:, i][0][j + 1][0])
    return dataset


data_arr = get_data()
print(data_arr[:5])
