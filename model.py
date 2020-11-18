import os
import tensorflow as tf
import scipy.io as sio
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras import layers

seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)

num_ch_c1 = 50
num_ch_c2 = 60

epochs = 1000
batch_size = 128
learning_rate = 0.001
optimizer_ = 'SGD'
use_dropout = False


# This is required when using GPU
def enable_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), " Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


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
    print('{:>6} images in total'.format(len(data_labels)))
    images_rgb = np.array(images_rgb, dtype=object)
    print('images loaded!')
    return images_rgb, data_labels


def make_car_make_prediction_model(num_ch_c1, num_ch_c2, use_dropout=False):
    model = tf.keras.Sequential()
    # model.add(layers.Input(shape=(3072,)))
    # model.add(layers.Reshape(target_shape=(32, 32, 3), input_shape=(3072,)))
    # model.add(
    #     layers.Conv2D(num_ch_c1, 9, activation='relu', padding="VALID", use_bias=True, input_shape=(32, 32, 3)))
    # model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='VALID', input_shape=(24, 24, 50)))
    # model.add(layers.Conv2D(num_ch_c2, 5, activation='relu', padding="VALID", use_bias=True, input_shape=(12, 12, 50)))
    # model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='VALID', input_shape=(8, 8, 60)))
    # model.add(layers.Flatten())
    # model.add(layers.Dense(300, use_bias=True))
    # model.add(layers.Dense(10, use_bias=True))
    return model


def main():
    model = make_car_make_prediction_model(num_ch_c1, num_ch_c2, use_dropout)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    if optimizer_ == 'SGD':
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_ == 'SGD-momentum':  # Question 3(a)
        raise NotImplementedError('Complete it by yourself')
    elif optimizer_ == 'RMSProp':  # Question 3(b)
        raise NotImplementedError('Complete it by yourself')
    elif optimizer_ == 'Adam':  # Question 3(c)
        raise NotImplementedError('Complete it by yourself')
    else:
        raise NotImplementedError(f'You do not need to handle [{optimizer_}] in this project.')

    images, labels = load_data()
    # print(images)
    # print(labels)

    if not os.path.exists('./models'):
        os.mkdir('./models')
    if not os.path.exists('./results'):
        os.mkdir('./results')

    # model.save(f'./models/car_make_prediction')


if __name__ == '__main__':
    enable_gpu()
    main()
