from sigmoid import sigmoid
import numpy as np


def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network


def identity_function(x):
    return x


def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    print(f"x: {x}\nW1: {W1}\nb1: {b1}\n")
    print("x * W1 + b1 = a1")
    print(f"{x} * {W1} + {b1} = {a1}")
    print(f"sigmoid(a1) = z1 = {z1}\n\n")

    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    print(f"z1: {z1}\nW2: {W2}\nb2: {b2}\n")
    print("z1 * W2 + b2 = a2")
    print(f"{z1} * {W2} + {b2} = {a2}")
    print(f"sigmoid(a2) = z2 = {z2}\n\n")

    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)
    print(f"z2: {z2}\nW3: {W3}\nb3: {b3}\n")
    print("z2 * W3 + b3 = a3")
    print(f"{z2} * {W3} + {b3} = {a3}\n\n")

    return y


network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)
