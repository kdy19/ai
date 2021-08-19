import matplotlib.pyplot as plt
import numpy as np


def relu(x):
    return np.maximum(0, x)


x = np.arange(-6.0, 6.0, 0.1)
y = relu(x)
plt.plot(x, y)
plt.ylim(-1.1, 6.1)
plt.show()