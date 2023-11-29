import tensorflow as tf
import numpy as np
from utils import *
from model.model import SNN
from model.snn_layer import heaveside
from pretreatment.dataset import load_mnist
import matplotlib.pyplot as plt

tf.random.set_seed(1)

def mean_loss(y_true, y_pred):
    one_hot = tf.one_hot(y_true, depth=10)
    mul = -y_pred * one_hot
    res = tf.reduce_mean(mul)
    return res
def test(data, model):
    res = 0
    k = 0
    for i in data:
        train = i["inputs"]
        prediction = model.predict(train).numpy()
        true = i["outputs"].numpy()
        res += np.sum(prediction == true)
        k += prediction.shape[0]
    print("accuracy on data is :", res / k)
    
def main():
    train_data, test_data = load_mnist()

    time_step = 1e-3
    tau_mem = 10e-3
    tau_syn = 5e-3
    alpha = np.exp(-time_step / tau_syn)
    beta = np.exp(-time_step / tau_mem)

    model = SNN(
            1,
            THRESHOLD,
            heaveside,
            [128, 10],
            [alpha],
            [beta]
            )
    # optimizer
    optimizer = tf.keras.optimizers.Adam()

    # compile the model with a custom loss
    model.compile(optimizer, mean_loss)
    model.fit(train_data, epochs = EPOCH)
    test(test_data, model)
    

