import numpy as np

class Layer:
    def __init__(self, neuronsCurrent, neuronsNext = 0, xavierInit = False):
        self.neurons = np.array(neuronsCurrent)
        self.length = neuronsCurrent

        if neuronsNext != 0:
            self.randomRange = 0.2
            self.weights = np.random.uniform(-self.randomRange,self.randomRange,(neuronsNext,neuronsCurrent))
            self.biases = np.random.uniform(-self.randomRange,self.randomRange,neuronsNext)

            # only for gradient descent
            self.lastWeightsDelta = np.zeros((neuronsCurrent,neuronsNext))
            self.lastBiasesDelta = np.zeros(neuronsNext)

            # xavier initialization
            if xavierInit:
                self.weights = np.random.randn(neuronsNext, neuronsCurrent).astype(np.float32) * np.sqrt(1.0/(neuronsNext))
                self.biases = np.zeros(neuronsNext)