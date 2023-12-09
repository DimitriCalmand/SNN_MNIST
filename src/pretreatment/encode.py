import tensorflow as tf
import numpy as np
from utils import *

def __encode(data):
    return data
    res = [data[:, :, 0]]
    for i in range(1, data.shape[-1]):
        mask = tf.cast(tf.math.not_equal(data[:, :, i-1], data[:, :, i]), "float32")
        res.append(mask)
    del data
    res = tf.stack(res, axis = 2)
    return res 
def encode(data, y, nb_batch):
    """ data shape (batch, y, x) """
    batch = data.shape[0]
    if (batch % nb_batch != 0):
        X = tf.reshape(data, (1, -1, batch))
        y = tf.reshape(y, (1, batch))
    else:
        time_step = batch // nb_batch
        X = tf.reshape(data, (nb_batch, -1, time_step))
        y = tf.reshape(y, (nb_batch, time_step))
    return [__encode(X), y]
    

