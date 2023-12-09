import tensorflow as tf
from utils import *

class metric_snn(tf.keras.metrics.Metric):
    def __init__(self, name = 'snn_metric', **kwargs):
        super(metric_snn, self).__init__(name = name, **kwargs)
        self.res = self.add_weight(name='res', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred):
        y_pred = tf.argmax(y_pred, axis = 1)
        y_true = tf.cast(y_true, "int64")
        res = tf.cast(y_pred == y_true, "float32")
        self.res.assign_add(tf.math.reduce_sum(res))
        size = tf.cast(tf.shape(y_true)[0] * tf.shape(y_true)[-1], tf.float32)
        self.count.assign_add(size)

    def result(self):
        return self.res / self.count
