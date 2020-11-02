import os
import scipy.io as sio
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def load_data():
    print('loading images ... ')
    mat = sio.loadmat('./data/cars_annos.mat')
    annotations = mat["annotations"]
    _, total_size = annotations.shape
    images_rgb = []
    data_labels = np.zeros((total_size, 6))
    for i in range(total_size):
        image_path = annotations[:, i][0][0][0]
        image_rgb = mpimg.imread("./data/" + image_path)
        images_rgb.append(image_rgb)
        for j in range(6):
            data_labels[i, j] = int(annotations[:, i][0][j + 1][0])
        if (i + 1) % 1000 == 0:
            print('{:>6} images loaded'.format(str(i + 1)))
        if i == 999:
            break
    images_rgb = np.array(images_rgb, dtype=object)
    print('images loaded!')
    return images_rgb, data_labels


images, labels = load_data()
print(images)
print(labels)
