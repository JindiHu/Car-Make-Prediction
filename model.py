import os
import tensorflow as tf
import scipy.io as sio
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
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
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    count = 1
    for i in range(total_size):
        bbox_x1 = annotations[:, i][0][1][0][0]
        bbox_y1 = annotations[:, i][0][2][0][0]
        bbox_x2 = annotations[:, i][0][3][0][0]
        bbox_y2 = annotations[:, i][0][4][0][0]
        class_label = annotations[:, i][0][5][0][0]
        is_test = annotations[:, i][0][6][0][0]
        # print(bbox_x1, bbox_y1, bbox_x2, bbox_y2)
        image_path = annotations[:, i][0][0][0]
        img = tf.keras.preprocessing.image.load_img("./data/" + image_path)
        img = tf.keras.preprocessing.image.img_to_array(img) / 255
        img = tf.image.crop_to_bounding_box(img, offset_height=bbox_y1, offset_width=bbox_x1,
                                            target_height=(bbox_y2 - bbox_y1), target_width=(bbox_x2 - bbox_x1))
        # img = tf.image.resize_with_pad(img, target_height=128, target_width=128, antialias=True)
        img = tf.image.resize(img, size=(128, 128), antialias=True)
        img = tf.make_tensor_proto(img)
        img = tf.make_ndarray(img)
        # print(img)
        if is_test == 0:
            train_x.append(img)
            train_y.append(class_label)
        else:
            test_x.append(img)
            test_y.append(class_label)
        if count % 1000 == 0:
            print('{:>6} images loaded'.format(str(count)))
        if count == 1000:
            break
        count += 1
    print('{:>6} images for training'.format(len(train_x)))
    print('{:>6} images for testing'.format(len(test_x)))
    # train_x.numpy()
    train_x = tf.convert_to_tensor(train_x, dtype=tf.dtypes.float32)
    train_y = tf.convert_to_tensor(train_y)
    train_y = train_y - 1
    test_x = tf.convert_to_tensor(test_x, dtype=tf.dtypes.float32)
    test_y = tf.convert_to_tensor(test_y)
    test_y = test_y - 1
    print('images loaded!')
    return train_x, train_y, test_x, test_y


def make_car_make_prediction_model(num_ch_c1, num_ch_c2, use_dropout=False):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(num_ch_c1, 3, activation='relu', padding="VALID", use_bias=True))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='VALID'))
    model.add(layers.Conv2D(num_ch_c2, 5, activation='relu', padding="VALID", use_bias=True))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='VALID'))
    model.add(layers.Flatten())
    model.add(layers.Dense(300, use_bias=True))
    model.add(layers.Dense(196, use_bias=True))
    return model


def main():
    model = make_car_make_prediction_model(num_ch_c1, num_ch_c2, use_dropout)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    train_x, train_y, test_x, test_y = load_data()
    # print(train_x[45])
    plt.imshow(train_x[0])
    print(train_x.shape)
    print(train_y)
    print(train_y.shape)
    plt.show()
    # print(images.shape)

    # print(labels.shape)

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

    model.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
    history = model.fit(
        train_x,
        train_y,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(test_x, test_y))
    print(history)

    if not os.path.exists('./models'):
        os.mkdir('./models')
    if not os.path.exists('./results'):
        os.mkdir('./results')

    # model.save(f'./models/car_make_prediction')


if __name__ == '__main__':
    enable_gpu()
    main()
