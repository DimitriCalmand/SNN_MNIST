import tensorflow as tf
import numpy as np

from model.snn_layer import snn_layer
from model.output_layer import output_layer


class SNN(tf.keras.Model):
    def __init__(self,
                nb_layer,
                threshold,
                spiking_function,
                units = [],
                alpha = [],
                beta = [],
            ):
        """if the size of the list < nb_layer then, all layer will take the 
        first argument,
        the nb_layer don't take in account the output layer wich will always
        be the -1 parameter of units, alpha, beta"""
        super(SNN, self).__init__()
        self.layer = []
        for i in range(nb_layer):
            u, a, b = self._select_good_args(nb_layer, units, alpha, beta)
            self.layer.append(snn_layer(u, a, b,
                threshold, spiking_function)
                )
        self.layer.append(output_layer(units[-1], alpha[-1], beta[-1]))
        # other param
        self.loss_metric = tf.keras.metrics.Mean(name="loss")

        self.accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(
                name="accuracy"
                ) 

    def _select_good_args(self, nb_layer, units, alpha, beta):
        if nb_layer <= len(units):
            u = units[0]
        else:
            u = units[i]
        if nb_layer <= len(alpha):
            a = alpha[0]
        else:
            a = alpha[i]
        if nb_layer <= len(beta):
            b = beta[0]
        else:
            b = beta[i]
        return u, a, b

    def call(self, inputs):
        for layer in self.layer:
            inputs = layer(inputs)
        return inputs

    def predict(self, inputs):
        prediction = self(inputs)
        prediction = tf.argmax(prediction, axis = 1)
        return prediction

    def train_step(self, inputs):
        X, y_true = inputs["inputs"], inputs["outputs"]
        with tf.GradientTape() as tape:
            prediction = self(X)
            # loss function
            loss = self.loss(y_true, prediction)
        gradient_tape = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(
            gradient_tape, self.trainable_variables)
            )
        self.loss_metric.update_state(loss)

        self.accuracy_metric.update_state(y_true, prediction)
        return {"loss":self.loss_metric.result(), "accuracy": self.accuracy_metric.result() }
