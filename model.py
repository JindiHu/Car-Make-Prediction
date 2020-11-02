import scipy.io as sio

mat = sio.loadmat('data/cars_annos.mat')

print(mat['annotations'])
