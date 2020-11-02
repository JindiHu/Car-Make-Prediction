import os
import scipy.io as sio
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


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


def peek_image(path, idx, labels):
    image_names = os.listdir("./data/" + path)
    img = mpimg.imread("./data/" + path + "/" + image_names[idx])
    print("image is", image_names[idx])
    name = image_names[idx].split('.')
    print("the label is " + str(labels[int(name[0]) - 1, 4]))
    plt.imshow(img)
    plt.show()


data_arr = get_data()
peek_image("car_ims", 0, data_arr)
print(data_arr[:5])
