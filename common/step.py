import matplotlib.pyplot as plt
import numpy as np

'''
def step_function(x):
    if x > 0:
        return 1
    else:
        return 0
'''


def step_function(x):
    y = x > 0

    print(type(y), y)
    return y.astype(np.int)


x = np.array([-1.0, 1.0, 2.0])
y = step_function(x)

print(type(y), y)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()
