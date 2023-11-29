import tensorflow as tf
import numpy as np

def mean_loss(y_true, y_pred):
    one_hot = tf.one_hot(y_true, depth=10)
    mul = -y_pred * one_hot
    res = tf.reduce_mean(mul)
    return res

