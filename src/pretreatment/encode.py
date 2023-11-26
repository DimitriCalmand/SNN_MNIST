import numpy as np

def encode(data : np.ndarray):
    # Data is an image that represent the digit
    res = np.unpackbits(data, axis=1)
    return res.reshape((data.shape[0] * data.shape[1], 8))

