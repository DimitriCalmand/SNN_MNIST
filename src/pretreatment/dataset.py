import tensorflow as tf

from utils import *
from pretreatment.encode import encode

def treat_dataset(X, y):
    def map_dataset(X, y):
        return {
                "inputs": X,
                "outputs": y
                }
    X = tf.reshape(X.astype("float32"), (tf.shape(X)[0], -1, 1)) / 255.
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(map_dataset)
    dataset = dataset.shuffle(SHUFFLE_SIZE).prefetch(16).cache()
    return dataset

def load_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    train = treat_dataset(x_train, y_train)
    test = treat_dataset(x_test, y_test)
    return train, test 
