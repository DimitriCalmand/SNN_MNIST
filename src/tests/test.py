from pretreatment.encode import encode
from pretreatment.dataset import load_mnist
import numpy as np
import tensorflow as tf
from model.model import SNN
from utils import *
from model.snn_layer import heaveside

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

def hyperparam():
    train_data, test_data = load_mnist()
    optimizer = tf.keras.optimizers.Adam()

    model1 = SNN(1, THRESHOLD, heaveside, [128, 10], [ALPHA], [BETA])
    model1.compile(optimizer, mean_loss)
    model1.fit(train_data, epochs = EPOCH, verbose = 0)
    test(test_data, model1)


def main():
    hyperparam()

if __name__ == "__main__":
    main()

