import numpy as np

class Layer:
    def __init__(self, neuronsCurrent, neuronsNext = 0):
        self.neurons = np.array(neuronsCurrent)

        if neuronsNext != 0:
            self.randomRange = 0.1
            self.weights = np.random.uniform(-self.randomRange,self.randomRange,(neuronsNext,neuronsCurrent))
            self.biases = np.random.uniform(-self.randomRange,self.randomRange,neuronsNext)
