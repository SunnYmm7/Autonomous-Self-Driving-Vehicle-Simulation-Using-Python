"""
ai/neural_network.py – Feed-forward neural network with bias terms.

Architecture:  SENSOR_COUNT (7) → HIDDEN_NODES (10) → 2 outputs
Output: [steering, acceleration] each in [-1, 1]

Weight layout in the flat genome vector:
  [W1: input×hidden | b1: hidden | W2: hidden×output | b2: output]

This compact layout allows genetic algorithm operations on a single 1D array.
"""
import numpy as np
from config import SENSOR_COUNT, HIDDEN_NODES


class NeuralNetwork:
    """
    Two-layer feed-forward neural network with tanh activation.
    
    Uses a flat weight vector for genetic algorithm compatibility.
    """

    def __init__(self):
        """Initialize network architecture and pre-compute weight slice indices."""
        self.n_in = SENSOR_COUNT  # 7 sensor inputs
        self.n_h = HIDDEN_NODES  # 10 hidden neurons
        self.n_out = 2  # 2 outputs: steering, acceleration

        # Pre-compute slice indices into the flat genome vector
        # This avoids recalculating on every forward pass
        self._w1_end = self.n_in * self.n_h
        self._b1_end = self._w1_end + self.n_h
        self._w2_end = self._b1_end + self.n_h * self.n_out
        self._b2_end = self._w2_end + self.n_out  # == GENOME_SIZE

    def predict(
        self, inputs: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """
        Forward pass: compute network output for given inputs and weights.

        Args:
            inputs: 1D array of shape (SENSOR_COUNT,)
                   Values typically in [0, 1] (normalized sensor distances)
            weights: Flat genome vector of length GENOME_SIZE

        Returns:
            2D array [steering, acceleration]
            Each value in [-1, 1] (tanh activation range)
        """
        # Extract weights from the flat genome
        w1 = weights[: self._w1_end].reshape(self.n_in, self.n_h)
        b1 = weights[self._w1_end : self._b1_end]
        w2 = weights[self._b1_end : self._w2_end].reshape(self.n_h, self.n_out)
        b2 = weights[self._w2_end : self._b2_end]

        # Layer 1: Input → Hidden with tanh activation
        hidden = np.tanh(inputs @ w1 + b1)

        # Layer 2: Hidden → Output with tanh activation
        output = np.tanh(hidden @ w2 + b2)

        return output