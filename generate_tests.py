import os
from scipy.signal import convolve
import random
import numpy as np
import json
from json import JSONEncoder

def gen_matrix(size, rand_range=(-100, 100)):
    mat = [[random.uniform(rand_range[0], rand_range[1]) for y in range(size[1])] for x in range(size[0])]
    return np.array(mat, dtype=float)

def write_test(f_name, mat, ker):
    class NumpyArrayEncoder(JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return JSONEncoder.default(self, obj)
        
    conv = convolve(mat, ker, mode='valid')
    ker = np.flip(ker, axis=0)
    ker = np.flip(ker, axis=1)
    data = {'matrix':{'shape':mat.shape, 'data':mat}, 'kernel':{'shape':ker.shape, 'data':ker}, 'convolution':{'shape':conv.shape, 'data':conv}}
    with open(f_name, 'w') as f:
        json.dump(data, f, cls=NumpyArrayEncoder)
    
generations = [{'fname':'test_0.json', 'm_size':(5,5), 'k_size':(2,2)},
               {'fname':'test_1.json', 'm_size':(10,10), 'k_size':(3,3)},
               {'fname':'test_2.json', 'm_size':(100,100), 'k_size':(10,10)},
               {'fname':'test_3.json', 'm_size':(320,240), 'k_size':(5,5)},
               {'fname':'test_4.json', 'm_size':(720,576), 'k_size':(7,7)},
               {'fname':'test_5.json', 'm_size':(1024,768), 'k_size':(15,15)} ]

if __name__ == '__main__':
    key_dir = 'keys'
    if key_dir not in os.listdir('./'):
        os.mkdir(key_dir)
    key_files = os.listdir(key_dir)
    
    for gen in generations:
        if gen['fname'] not in key_files:
            mat    = gen_matrix(gen['m_size']) 
            kernel = gen_matrix(gen['k_size'])
            f_name = '{}/{}'.format(key_dir, gen['fname'])
            write_test(f_name, mat, kernel)
            print(f'file written: {f_name}')
    
    
    

