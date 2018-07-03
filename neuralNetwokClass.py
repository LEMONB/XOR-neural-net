import numpy as np
import nnLayerClass as nl
import math
import datetime

class Network:
    def __init__(self, inputNodes, hiddenLayersAndNodes, outputNodes, xavierInit = False):
        self.inputLayer = nl.Layer(inputNodes,hiddenLayersAndNodes[0],xavierInit)

        self.hiddenLayers = np.empty(len(hiddenLayersAndNodes),dtype=nl.Layer)
        for i in range(len(hiddenLayersAndNodes)):
            if i == len(hiddenLayersAndNodes)-1:
                self.hiddenLayers[i] = nl.Layer(hiddenLayersAndNodes[i], outputNodes, xavierInit)
            else:
                self.hiddenLayers[i] = nl.Layer(hiddenLayersAndNodes[i], hiddenLayersAndNodes[i + 1], xavierInit)

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

    def train(self, inputData, expectedData, learningRate = 0.7, inertia = 0.3):
        prediction = self.feedForward(inputData)

        errors = expectedData - prediction
        neuronsDelta = errors * np.full(self.outputLayer.length,(1 - self.outputLayer.neurons) * self.outputLayer.neurons)

        # weights inside hidden layers
        for i in range(len(self.hiddenLayers)-1,-1,-1):
            if i == len(self.hiddenLayers)-1:
                grad = np.dot(np.transpose(np.reshape(self.hiddenLayers[i].neurons, (1, self.hiddenLayers[i].length))), np.reshape(neuronsDelta, (1, self.outputLayer.length)))
            else:
                grad = np.dot(np.transpose(np.reshape(self.hiddenLayers[i].neurons, (1, self.hiddenLayers[i].length))), np.reshape(neuronsDelta, (1, self.hiddenLayers[i+1].length)))

            weightsDelta = grad * learningRate + self.hiddenLayers[i].lastWeightsDelta * inertia
            self.hiddenLayers[i].lastWeightsDelta = weightsDelta

            biasesDelta = neuronsDelta * learningRate + self.hiddenLayers[i].lastBiasesDelta * inertia
            self.hiddenLayers[i].lastBiasesDelta = biasesDelta

            self.hiddenLayers[i].weights += np.transpose(weightsDelta)
            self.hiddenLayers[i].biases += biasesDelta

            neuronsDelta = np.dot(np.transpose(self.hiddenLayers[i].weights), neuronsDelta) * \
                              np.full(self.hiddenLayers[i].length,(1 - self.hiddenLayers[i].neurons) * self.hiddenLayers[i].neurons)

        # weights from input layer
        grad = np.dot(np.transpose(np.reshape(self.inputLayer.neurons, (1, self.inputLayer.length))), np.reshape(neuronsDelta, (1, self.hiddenLayers[0].length)))

        weightsDelta = grad * learningRate + self.inputLayer.lastWeightsDelta * inertia
        self.inputLayer.lastWeightsDelta = weightsDelta

        biasesDelta = neuronsDelta * learningRate + self.inputLayer.lastBiasesDelta * inertia
        self.inputLayer.lastBiasesDelta = biasesDelta

        self.inputLayer.weights += np.transpose(weightsDelta)
        self.inputLayer.biases += biasesDelta

    def sigmoid(self, x):
        return 1/(1+math.exp(-x))

    def save(self):
        print()
        timeStamp = str(datetime.datetime.now())
        fileName = timeStamp
        fileName = fileName[:-7]
        fileName += str(self.inputLayer.length) + str(len(self.hiddenLayers)) + str(self.outputLayer.length) + ".txt"
        fileName = fileName.replace(" ", "")
        fileName = fileName.replace("-", "")
        fileName = fileName.replace(":", "")
        file = open(fileName, "w+")

        for i in range(self.hiddenLayers[0].length):
            for j in range(self.inputLayer.length):
                file.write(str(self.inputLayer.weights[i][j]) + "\n")
        file.write("---\n")

        for i in range(self.hiddenLayers[0].length):
            file.write(str(self.inputLayer.biases[i]) + "\n")
        file.write("---\n")

        for i in range(len(self.hiddenLayers)-1):
            for j in range(self.hiddenLayers[i+1].length):
                for k in range(self.hiddenLayers[i].length):
                    file.write(str(self.hiddenLayers[i].weights[j][k]) + "\n")
        file.write("---\n")

        for i in range(len(self.hiddenLayers)-1):
            for j in range(self.hiddenLayers[i+1].length):
                file.write(str(self.hiddenLayers[i].biases[j]) + "\n")
        file.write("---\n")

        for i in range(self.outputLayer.length):
            for j in range(self.hiddenLayers[len(self.hiddenLayers)-1].length):
                file.write(str(self.hiddenLayers[len(self.hiddenLayers)-1].weights[i][j]) + "\n")
        file.write("---\n")

        for i in range(self.outputLayer.length):
            file.write(str(self.hiddenLayers[len(self.hiddenLayers)-1].biases[i]) + "\n")
        file.write("---\n")

        file.close()
        print("NN saved")

    def load(self, fileName):
        if (str(self.inputLayer.length) + str(len(self.hiddenLayers)) + str(self.outputLayer.length)) in fileName:
            file = open(fileName,"r")

            for i in range(self.hiddenLayers[0].length):
                for j in range(self.inputLayer.length):
                    self.inputLayer.weights[i][j] = np.float32(file.readline())
            file.readline()

            for i in range(self.hiddenLayers[0].length):
                self.inputLayer.biases[i] = np.float32(file.readline())
            file.readline()

            for i in range(len(self.hiddenLayers) - 1):
                for j in range(self.hiddenLayers[i + 1].length):
                    for k in range(self.hiddenLayers[i].length):
                        self.hiddenLayers[i].weights[j][k] = np.float32(file.readline())
            file.readline()

            for i in range(len(self.hiddenLayers) - 1):
                for j in range(self.hiddenLayers[i + 1].length):
                    self.hiddenLayers[i].biases[j] = np.float32(file.readline())
            file.readline()

            for i in range(self.outputLayer.length):
                for j in range(self.hiddenLayers[len(self.hiddenLayers) - 1].length):
                    self.hiddenLayers[len(self.hiddenLayers) - 1].weights[i][j] = np.float32(file.readline())
            file.readline()

            for i in range(self.outputLayer.length):
                self.hiddenLayers[len(self.hiddenLayers) - 1].biases[i] = np.float32(file.readline())
            file.readline()

            file.close()

            print("NN loaded")