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
        batch = tf.shape(inputs)[0]
        time_step = tf.shape(inputs)[2]
        h = tf.einsum("abc,bd->acd", inputs, self.weights)
        membrane_potential = tf.zeros((batch, time_step))
        synapse = tf.zeros((batch, time_step))
        outputs = [] 
        for t in range(time_step):
            s_t = spiking_function(membrane_potential - self.threshold)
            i_t_1 = self.alpha * synapse + h[:, t]
            v_t_1 = (self.beta * membrane_potential + synapse) * (self.threshold - s_t)
            outputs.append(s_t)
            membrane_potential = v_t_1
            synapse = s_t
        return outputs
