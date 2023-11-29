import tensorflow as tf
import numpy as np
from utils import *
from model.model import SNN
from model.snn_layer import heaveside
from pretreatment.dataset import load_mnist
import matplotlib.pyplot as plt
from model.loss import *


def test(data, model):
    # function that print the accuracy
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
    # simple snn
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
    # test the model
    test(test_data, model)
    

