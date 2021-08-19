import numpy as np


def softmax(x):
    max_x = np.max(x)
    exp_x = np.exp(x - max_x)
    sum_exp_x = np.sum(exp_x)
    
    return exp_x / sum_exp_x


if __name__ == '__main__':
    x = np.array([0.3, 2.9, 4.0])
    print(softmax(x))
