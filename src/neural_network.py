# %% Run this code to test the neural network
import nnfs
from nnfs.datasets import spiral_data
from src.utils import create_data, Layer_Dense, Activation_ReLU


nnfs.init()

X, y = spiral_data(100, 3)

layer1 = Layer_Dense(2, 5)
activation1 = Activation_ReLU()
layer1.forward(X)
activation1.forward(layer1.output)
print(activation1.output)


