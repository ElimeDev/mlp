from mlp import *
import numpy as np

x = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])
y = np.array([
    [0],
    [1],
    [1],
    [0],
])

nb_iteration = 2000
learning_rate = 2

mlp = MLP([2, 2, 1])
losses = mlp.train(x, y, nb_iteration= nb_iteration, learning_rate= learning_rate)

print("final mse : ", losses[-1])