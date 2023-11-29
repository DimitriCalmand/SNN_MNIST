import tensorflow as tf
import numpy as np

class output_layer(tf.keras.layers.Layer):
    def __init__(self,
            units,
            alpha,
            beta
        ):
        super(output_layer, self).__init__()
        self.units = units
        self.alpha = alpha
        self.beta = beta
    def build(self, input_shape):
        self.w = self.add_weight(
                shape = (input_shape[-2], self.units),
                trainable = True
                )
    def call(self, inputs):
        batch = tf.shape(inputs)[0]
        time_step = inputs.shape[2]
        h = tf.einsum("abc,bd->acd", inputs, self.w)
        membrane_potential = tf.zeros((batch, self.units))
        synapse = tf.zeros((batch, self.units))

        def condition(t, *args):
            return t < time_step

        def body(t, membrane_potential, synapse, outputs):
            i = self.alpha * synapse + h[:, t]
            v = self.beta * membrane_potential + i
            outputs = outputs.write(t, v)
            return t + 1, v, i, outputs 

        t_final, _, _, outputs = tf.while_loop(condition,
                body,
                (
                    0, membrane_potential,
                    synapse,
                    tf.TensorArray(dtype=tf.float32,
                        size=time_step,
                        )
                    )
                )
        outputs = outputs.stack() 
        outputs = tf.transpose(outputs, perm = [1, 2, 0])
        m = tf.reduce_max(outputs, axis=-1)
        log_p_y = tf.nn.log_softmax(m)
        return log_p_y 
