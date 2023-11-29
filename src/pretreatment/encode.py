import tensorflow as tf
import numpy as np

def encode(data):
    data = data.numpy()
    res = np.unpackbits(data, axis=1)
    res = res.reshape((data.shape[0], data.shape[1], 8))
    return res.astype(np.float32)
def encode_y(y):
    y = y.numpy()
    res = np.unpackbits(data, axis=1)
    res = res.reshape((data.shape[0], data.shape[1], 8))
    
