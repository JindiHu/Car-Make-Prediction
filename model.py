import os
import scipy.io as sio
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def load_data():
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


def peek_image(path, idx, dataset):
    files = os.listdir("./data/" + path)
    img = mpimg.imread("./data/" + path + "/" + files[idx])
    print("image is", files[idx])
    img_name, extension = files[idx].split('.')
    print(img_name)
    print("the label is " + str(dataset[int(img_name[0]) - 1, 4]))
    plt.imshow(img)
    plt.show()


data_arr = load_data()
peek_image("car_ims", 0, data_arr)
print(data_arr[:5])
