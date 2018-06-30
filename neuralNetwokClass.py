import numpy as np
import nnLayerClass as nl
import math

class Network:
    def __init__(self,inputNodes,hiddenLayersAndNodes,outputNodes):
        self.inputLayer = nl.Layer(inputNodes,hiddenLayersAndNodes[0])

        self.hiddenLayers = np.array((1, len(hiddenLayersAndNodes)),dtype=nl.Layer)
        for i in range(len(hiddenLayersAndNodes)):
            if i == len(hiddenLayersAndNodes)-1:
                self.hiddenLayers[i] = nl.Layer(hiddenLayersAndNodes[i], outputNodes)
            else:
                self.hiddenLayers[i] = nl.Layer(hiddenLayersAndNodes[i], hiddenLayersAndNodes[i + 1])

        self.outputLayer = nl.Layer(outputNodes)

    def feedForward(self, input):
        self.inputLayer.neurons = input
        activation = np.vectorize(self.sigmoid)

        for i in range(len(self.hiddenLayers)):
            if i == 0:  # first hidden layer
                self.hiddenLayers[i].neurons = np.dot(self.inputLayer.weights, self.inputLayer.neurons) + self.inputLayer.biases
            else: # else hidden layer
                self.hiddenLayers[i].neurons = np.dot(self.hiddenLayers[i-1].weights, self.hiddenLayers[i - 1].neurons) + self.hiddenLayers[i-1].biases

            self.hiddenLayers[i].neurons = activation(self.hiddenLayers[i].neurons)

        # output layer
        self.outputLayer.neurons = np.dot(self.hiddenLayers[len(self.hiddenLayers)-1].weights,
                                          self.hiddenLayers[len(self.hiddenLayers) - 1].neurons) \
                                   + self.hiddenLayers[len(self.hiddenLayers)-1].biases
        self.outputLayer.neurons = activation(self.outputLayer.neurons)

        return self.outputLayer.neurons

    def train(self, inputData, expectedData, learningRate = 0.3, inertia = 0.3):
        prediction = self.feedForward(inputData)

        error = expectedData - prediction

        # weights from hidden to output
        deltaOut = error * np.full(len(self.outputLayer.neurons),
                                   (1 - self.outputLayer.neurons) * self.outputLayer.neurons)
        for i in range(len(self.hiddenLayers)-1,0):
            if i == len(self.hiddenLayers)-1:
                hiddenDelta = np.dot(np.transpose(self.hiddenLayers[i].weights), deltaOut) * \
                              np.full(len(self.hiddenLayers[i].neurons),(1 - self.hiddenLayers[i].neurons) * self.hiddenLayers[i].neurons)

                grad_2 = np.dot(np.transpose(np.reshape(self.hiddenLayers[i].neurons, (1, len(self.hiddenLayers[i].neurons)))), np.reshape(deltaOut, (1, len(self.hiddenLayers[i].neurons))))

                deltaWeights_2 = grad_2 * learningRate + (lastDeltaWeights_2 * inertia)
                lastDeltaWeights_2 = deltaWeights_2

                deltaBiases_2 = deltaOut * learningRate + (lastDeltaBiases_2 * inertia)
                lastDeltaBiases_2 = deltaBiases_2

                weights_2 += np.transpose(deltaWeights_2)
                biases_2 += deltaBiases_2

        # weights from input to hidden
        grad_1 = np.dot(np.transpose(np.reshape(inputs, (1, inputNeurons))), np.reshape(hiddenDelta, (1, hiddenNeurons)))

        deltaWeights_1 = grad_1 * learningRate + (lastDeltaWeights_1 * inertia)
        lastDeltaWeights_1 = deltaWeights_1

        deltaBiases_1 = hiddenDelta * learningRate + (lastDeltaBiases_1 * inertia)
        lastDeltaBiases_1 = deltaBiases_1

        weights_1 += np.transpose(deltaWeights_1)
        biases_1 += deltaBiases_1

    def sigmoid(self, x):
        return 1/(1+math.exp(-x))