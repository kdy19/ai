import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x) -> float:
    return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    x = np.arange(-5.0, 5.0, 0.1) # 5.0 is except
    y = sigmoid(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()
