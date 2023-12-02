import tensorflow as tf
import numpy as np

def mean_loss(y_true, y_pred):
    one_hot = tf.one_hot(y_true, depth=10, axis = 1)
    mul = -y_pred * one_hot
    res = tf.reduce_mean(mul, axis = 0)
    return res
def bce(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    one_hot = tf.one_hot(y_true, depth=10, axis = 1)
    return bce(one_hot, y_pred)
    
