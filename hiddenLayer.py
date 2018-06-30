import numpy as np

class layer:
    def __init__(self, neuronsPrev, neuronsCurrent, output = False):
        neurons = np.array(neuronsCurrent)

        randomRange = 0.1
        weights = np.random.uniform(-randomRange,randomRange,(neuronsCurrent,neuronsPrev))
        if not output:
            biases = np.random.uniform(-randomRange,randomRange,neuronsCurrent)
