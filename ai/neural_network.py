import numpy as np
from config import SENSOR_COUNT, HIDDEN_NODES

class NeuralNetwork:
    def __init__(self):
        self.input_nodes = SENSOR_COUNT
        self.hidden_nodes = HIDDEN_NODES
        self.output_nodes = 2 # [Steering, Speed]

    def predict(self, inputs, weights):
        # Split flat weights into matrices
        split = self.input_nodes * self.hidden_nodes
        w1 = weights[:split].reshape(self.input_nodes, self.hidden_nodes)
        w2 = weights[split:].reshape(self.hidden_nodes, self.output_nodes)

        # Forward pass
        h = np.tanh(np.dot(inputs, w1))
        out = np.tanh(np.dot(h, w2))
        return out # returns values between -1 and 1