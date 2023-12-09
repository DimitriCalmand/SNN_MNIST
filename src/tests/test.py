from pretreatment.encode import encode, __encode
from pretreatment.dataset import load_mnist
import numpy as np
import tensorflow as tf
from model.model import SNN
from utils import *
from model.snn_layer import heaveside
from model.loss import *
from time import time

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

    print("model 1 ---------------------")
    tps = time()
    model1 = SNN(1, THRESHOLD, heaveside, [128, 10], [ALPHA], [BETA])
    model1.compile(optimizer, mean_loss)
    model1.fit(train_data, epochs = EPOCH, verbose = 0)
    test(test_data, model1)
    print("time = ", time() - tps)

    print("model 2 ---------------------")
    tps = time()
    optimizer = tf.keras.optimizers.Adam()
    model1 = SNN(1, THRESHOLD, heaveside, [256, 10], [ALPHA], [BETA])
    model1.compile(optimizer, mean_loss)
    model1.fit(train_data, epochs = EPOCH, verbose = 0)
    test(test_data, model1)
    print("time = ", time() - tps)

    print("model 3 ---------------------")
    tps = time()
    optimizer = tf.keras.optimizers.Adam()
    model1 = SNN(1, THRESHOLD, heaveside, [128, 10], [0.9], [BETA])
    model1.compile(optimizer, mean_loss)
    model1.fit(train_data, epochs = EPOCH, verbose = 0)
    test(test_data, model1)
    print("time = ", time() - tps)

    print("model 4 ---------------------")
    tps = time()
    optimizer = tf.keras.optimizers.Adam()
    model1 = SNN(1, THRESHOLD, heaveside, [128, 10], [0.9], [0.9])
    model1.compile(optimizer, mean_loss)
    model1.fit(train_data, epochs = EPOCH, verbose = 0)
    test(test_data, model1)
    print("time = ", time() - tps)

    print("model 5 ---------------------")
    tps = time()
    optimizer = tf.keras.optimizers.Adam()
    model1 = SNN(2, THRESHOLD, heaveside, [256, 256, 10], [ALPHA], [BETA])
    model1.compile(optimizer, mean_loss)
    model1.fit(train_data, epochs = EPOCH, verbose = 0)
    test(test_data, model1)
    print("time = ", time() - tps)

def test_encode():
    data = tf.ones((256, 28, 28))
    y = tf.ones((256))
    X, y = encode(data, y, 4)
    print(X.shape)
    data = tf.constant([[[1, 0, 1, 0], [0, 1, 0, 0]]], dtype = "float32")
    res = __encode(data)
    print(data)
    print(res)

def main():
    test_encode()
    #hyperparam()

if __name__ == "__main__":
    main()

