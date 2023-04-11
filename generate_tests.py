import os
from scipy.signal import convolve
import random
import numpy as np
import json
from json import JSONEncoder
import tensorflow as tf
from tensorflow.keras.layers import MaxPool3D

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def gen_matrix(size, rand_range=(-100, 100)):
    # generation of 2d
    if len(size) == 2:
        mat = [[random.uniform(rand_range[0], rand_range[1]) for y in range(size[1])] for x in range(size[0])]
    # handle the genration with multiple channels (3s)
    elif len(size) == 3:
        mat = [[[random.uniform(rand_range[0], rand_range[1]) for z in range(size[2])] for y in range(size[1])] for x in range(size[0])]
    return np.array(mat, dtype=float)

def write_test_conv(f_name, mat, ker):
    conv = convolve(mat, ker, mode='valid')
    ker = np.flip(ker, axis=0)
    ker = np.flip(ker, axis=1)
    ker = np.flip(ker, axis=2)
    data = {'matrix':{'shape':mat.shape, 'data':mat}, 'kernel':{'shape':ker.shape, 'data':ker}, 'convolution':{'shape':conv.shape, 'data':conv}}
    with open(f_name, 'w') as f:
        json.dump(data, f, cls=NumpyArrayEncoder)

def write_test_maxpool(f_name, mat, pool_size, stride):
    og_shape = mat.shape
    mat = tf.constant(mat)
    mat = tf.reshape(mat, [1, mat.shape[0], mat.shape[1], mat.shape[2], 1])
    pool = MaxPool3D(pool_size=(1, pool_size[0], pool_size[1]), strides=(1, stride, stride), padding='valid')
    
    output = pool(mat).numpy()
    output = np.reshape(output, [og_shape[0], 
                                 (og_shape[1] - pool_size[0]) // stride + 1, 
                                 (og_shape[2] - pool_size[1]) // stride + 1])
    mat = mat.numpy()
    mat = np.reshape(mat, og_shape)

    data = {'matrix':{'shape':mat.shape, 'data':mat}, 'pool':{'shape':pool_size, 'stride':stride}, 'output':{'shape':output.shape, 'data':output}}
    with open(f_name, 'w') as f:
        json.dump(data, f, cls=NumpyArrayEncoder)

def write_test_fullconn(f_name, mat, w_and_b):
    new_mat = mat.flatten()
    new_mat = np.vstack((new_mat, np.ones(new_mat.shape[0]))).T

    output = [np.sum(np.multiply(new_mat, w_and_b[0][:, i:i+2])) for i in range(0, w_and_b.shape[2], 2)]
    output = np.reshape(output, (1, w_and_b.shape[2]//2, 1))

    data = {'matrix':{'shape':mat.shape, 'data':mat}, 'w_and_b':{'shape':w_and_b.shape, 'data':w_and_b}, 'output':{'shape':output.shape, 'data':output}}
    with open(f_name, 'w') as f:
        json.dump(data, f, cls=NumpyArrayEncoder)

generations_conv = [{'fname':'test_conv_0.json', 'm_size':(1,5,5), 'k_size':(1,2,2)},
                    {'fname':'test_conv_1.json', 'm_size':(1,10,10), 'k_size':(1,3,3)},
                    {'fname':'test_conv_2.json', 'm_size':(1,100,100), 'k_size':(1,10,10)},
                    {'fname':'test_conv_3.json', 'm_size':(1,320,240), 'k_size':(1,5,5)},
                    {'fname':'test_conv_4.json', 'm_size':(1,720,576), 'k_size':(1,7,7)},
                    {'fname':'test_conv_5.json', 'm_size':(1,1024,768), 'k_size':(1,15,15)},
                    {'fname':'test_conv3D_0.json', 'm_size':(3,5,5), 'k_size':(1,2,2)},
                    {'fname':'test_conv3D_1.json', 'm_size':(3,10,10), 'k_size':(1,3,3)},
                    {'fname':'test_conv3D_2.json', 'm_size':(3,100,100), 'k_size':(1,10,10)},
                    {'fname':'test_conv3D_3.json', 'm_size':(3,320,240), 'k_size':(1,5,5)},
                    {'fname':'test_conv3D_4.json', 'm_size':(3,720,576), 'k_size':(1,7,7)},
                    {'fname':'test_conv3D_5.json', 'm_size':(3,1024,768), 'k_size':(1,15,15)},
                    {'fname':'test_conv3D_6.json', 'm_size':(384,15,15), 'k_size':(384,3,3)}]

generations_maxpool = [{'fname':'test_maxpool_0.json', 'm_size':(1,13,13),   'w_size':(3,3), 'stride':2},
                       {'fname':'test_maxpool_alexnet_3.json', 'm_size':(256,13,13), 'w_size':(3,3), 'stride':2},
                       {'fname':'test_maxpool_alexnet_1.json', 'm_size':(96,55,55), 'w_size':(3,3), 'stride':2}]

generations_fullconn = [{'fname':'test_fullconn_0.json', 'm_size':(1,2,2), 'wb_size':(1, 4, 4)},
                        {'fname':'test_fullconn_1.json', 'm_size':(256,6,6), 'wb_size':(1, 256*6*6, 4096*2)}]

if __name__ == '__main__':
    key_dir = 'keys'
    if key_dir not in os.listdir('./'):
        os.mkdir(key_dir)
    key_files = os.listdir(key_dir)

    for gen in generations_conv:
        if gen['fname'] not in key_files:
            mat    = gen_matrix(gen['m_size'])
            kernel = gen_matrix(gen['k_size'])
            f_name = '{}/{}'.format(key_dir, gen['fname'])
            write_test_conv(f_name, mat, kernel)
            print(f'file written: {f_name}')

    for gen in generations_maxpool:
        if gen['fname'] not in key_files:
            mat       = gen_matrix(gen['m_size'])
            pool_size = gen['w_size']
            stride    = gen['stride']
            f_name = '{}/{}'.format(key_dir, gen['fname'])

            write_test_maxpool(f_name, mat, pool_size, stride)
            print(f'file written: {f_name}')

    for gen in generations_fullconn:
        if gen['fname'] not in key_files:
            mat     = gen_matrix(gen['m_size'])
            w_and_b = gen_matrix(gen['wb_size'])
            f_name  = '{}/{}'.format(key_dir, gen['fname'])
            
            write_test_fullconn(f_name, mat, w_and_b)
            print(f'file written: {f_name}')
