import random
import math
import time
import pygame
import numpy as np

pygame.init()
screen = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()

def sigmoid(x):
    return 1/(1+math.exp(-x))

def drawNet(x):
    print(inputs[0],inputs[1])
    print(round(outputOutputs[0],2),"(",trainAnswers[x],")")
    pygame.display.update()

maxEpoch = 10000
learningRate = 0.7
moment = 0.3
bias = 1
inputNeurons = 2
hiddenNeurons = 5
outputNeurons = 1
trainSet = np.array([[0,0],[0,1],[1,0],[1,1]])
trainAnswers = np.array([[0],[1],[1],[0]])

randomRange = 0.2
inputs = trainSet[0] #[random.randint(0,1),random.randint(0,1)]
weights_1 = np.random.random((hiddenNeurons,inputNeurons))
weights_2 = np.random.random((outputNeurons,hiddenNeurons))
biases_1 = np.random.random(hiddenNeurons)
biases_2 = np.random.random(outputNeurons)
hiddenInputs = np.array(hiddenNeurons)
hiddenOutputs = np.array(hiddenNeurons)
outputInputs = np.array(outputNeurons)
outputOutputs = np.array(outputNeurons)

averageEpochError = 0
errors = np.zeros(outputNeurons) #[0] * len(trainAnswers)
deltaOut = np.zeros(outputNeurons)
deltasH = np.zeros(hiddenNeurons)
lastDeltaWeights_1 = np.zeros((inputNeurons,hiddenNeurons))
lastDeltaWeights_2 = np.zeros((hiddenNeurons,outputNeurons))
lastDeltaBiases_1 = np.zeros(hiddenNeurons)
lastDeltaBiases_2 = np.zeros(outputNeurons)

screen.fill((192, 192, 192))
clock.tick(60)

font = pygame.font.SysFont("comicsansms", 10)

pygame.display.update()

for j in range(0,maxEpoch):
    print("----------- epoch",j,"------")
    for i in range(0,4,1):
        print("--set",i,"--")
        inputs = trainSet[i]
        #print("in", inputs[0], inputs[1])

        # feed forward
        hiddenInputs = np.dot(weights_1, inputs) + biases_1
        activation = np.vectorize(sigmoid)
        hiddenOutputs = activation(hiddenInputs)

        outputInputs = np.dot(weights_2, hiddenOutputs) + biases_2
        outputOutputs = activation(outputInputs)
        # end of feed forward


        #print("out", outputOutputs)
        errors = trainAnswers[i] - outputOutputs
        #print("err",i, errors[i])

        # weights from hidden to output
        deltaOut = errors * np.full(outputNeurons, (1 - outputOutputs) * outputOutputs)
        deltasH = np.dot(np.transpose(weights_2), deltaOut) * np.full(hiddenNeurons, (1 - hiddenOutputs) * hiddenOutputs)

        grad_2 = np.dot(np.transpose(np.reshape(hiddenOutputs, (1, hiddenNeurons))), np.reshape(deltaOut, (1, outputNeurons)))

        deltaWeights_2 = grad_2 * learningRate + (lastDeltaWeights_2 * moment)
        lastDeltaWeights_2 = deltaWeights_2

        deltaBiases_2 = deltaOut * learningRate + (lastDeltaBiases_2 * moment)
        lastDeltaBiases_2 = deltaBiases_2

        weights_2 += np.transpose(deltaWeights_2)
        biases_2 += deltaBiases_2

        # weights from input to hidden
        grad_1 = np.dot(np.transpose(np.reshape(inputs, (1, inputNeurons))), np.reshape(deltasH, (1, hiddenNeurons)))

        deltaWeights_1 = grad_1 * learningRate + (lastDeltaWeights_1 * moment)
        lastDeltaWeights_1 = deltaWeights_1

        deltaBiases_1 = deltasH * learningRate + (lastDeltaBiases_1 * moment)
        lastDeltaBiases_1 = deltaBiases_1

        weights_1 += np.transpose(deltaWeights_1)
        biases_1 += deltaBiases_1

        # drawing network
        drawNet(i)
        #time.sleep(0.001)
        print()

        for err in errors:
            averageEpochError += math.pow(err,2)

    averageEpochError /= 4
    print("averageEpochError ", averageEpochError)
    if averageEpochError < 0.001:
        break
    #print("----------- epoch",j,"------")
    print()

dummy = input()