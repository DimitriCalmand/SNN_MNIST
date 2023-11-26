from pretreatment.encode import encode
import numpy as np

def test_encode():
    a = np.arange(0, 2 * 3).reshape((2, 3)).astype(np.uint8)
    b = encode(a)
    if b.shape != (6, 8):
        raise Exception("Probleme with the encoder")
def main():
    test_encode()
if __name__ == "__main__":
    main()

