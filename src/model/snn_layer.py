import tensorflow as tf
import numpy as np

from utils import *

@tf.custom_gradient
def heaveside(inputs):
    def grad(dy):
        return dy * (GRADIENT_SCALE * tf.square(1 / (tf.abs(inputs) + 1)))
    return tf.where(inputs > 0, tf.ones_like(inputs), tf.zeros_like(inputs)), grad

class snn_layer(tf.keras.layers.Layer):
    def __init__(self,
            units,
            alpha,
            beta,
            threshold,
            spiking_function
            ):
        super(snn_layer, self).__init__()
        self.units = units # number of neuron

        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold

        self.spiking_function = spiking_function # args = ()

    def build(self, input_shape):
        # inputs shape (batch, pixel, nb_bit)
        self.w = self.add_weight(
            shape=(input_shape[-2], self.units),
            trainable=True,
            )

    def call(self, inputs):
        # inputs shape (batch, pixel, nb_bit)
        batch = tf.shape(inputs)[0]
        time_step = inputs.shape[2]
        h = tf.einsum("abc,bd->acd", inputs, self.w)

        membrane_potential = tf.zeros((batch, self.units))
        synapse = tf.zeros((batch, self.units))
        prev_s = tf.zeros((batch, self.units))

        def condition(t, *args):
            # the condition of the tf.loop
            return t < time_step

        def body(t, membrane_potential, synapse, outputs, prev_s):
            h = tf.einsum("ab,bd->ad", inputs[:, :, t], self.w)
            i = self.alpha * synapse + h
            v = (self.beta * membrane_potential + i) * (self.threshold - prev_s)
            s = heaveside(v - self.threshold)
            rst = tf.stop_gradient(s)
            outputs = outputs.write(t, s)
            return t + 1, membrane_potential, synapse, outputs, prev_s 

        # I made a tf_loop because the gradient can't compute without it
        t_final, _, _, outputs, _ = tf.while_loop(condition,
                body,
                (
                    0, membrane_potential,
                    synapse,
                    tf.TensorArray(dtype=tf.float32,
                        size=time_step
                        ),
                    prev_s
                    )
                )
        outputs = outputs.stack()
        outputs = tf.transpose(outputs, perm = [1, 2, 0])
        return outputs
