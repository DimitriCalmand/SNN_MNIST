import tensorflow as tf

from utils import *
from pretreatment.encode import encode

def treat_dataset(X, y):
    def map_dataset(X, y):
        tmp = encode(X, y, BATCH_SIZE // 1)
        X = tmp[0]
        y = tmp[1]
        return {
                "inputs": X,
                "outputs": y
                }
    X = X.astype("float32")
    X = tf.where(X > 127, 255.00, 0.0)
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.batch(BATCH_SIZE, drop_remainder = True)
    dataset = dataset.map(map_dataset)
    dataset = dataset.shuffle(SHUFFLE_SIZE).prefetch(16).cache()
    return dataset

def load_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    train = treat_dataset(x_train, y_train)
    test = treat_dataset(x_test, y_test)
    return train, test 
