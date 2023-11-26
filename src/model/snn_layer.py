import tensorflow as tf
import numpy as np

class snn_layer(tf.keras.layers.Layer):
    def __init__(self,
            units,
            alpha,
            beta,
            threshold,
            spiking_function
            ):
        self.units = units
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold
        self.spiking_function = spiking_function # args = ()
    def build(self, input_shape):
        # inputs shape (batch, pixel, nb_bit)
        self.weights = self.add_weight(
            shape=(input_shape[-2], self.units),
            trainable=True,
            )
    def call(inputs):
        # inputs shape (batch, pixel, nb_bit)
        h = tf.einsum("abc,bd->acd", inputs, self.weights)
        for t in range(inputs.shape[-1]):
            spiking_function(h[t])

    
