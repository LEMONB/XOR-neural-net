import random
import math
import time
import pygame
import numpy as np
import neuralNetwokClass as network
import matplotlib.pyplot as plt

pygame.init()
screen = pygame.display.set_mode((1280, 800))
clock = pygame.time.Clock()

# MAIN METHODS
def drawNet(x = 0):
    screen.fill((192, 192, 192))
    distBetweenLayers = (RIGHT_BORDER - LEFT_BORDER) / (len(nn.hiddenLayers) + 1)

    # drawing weights_1
    for i in range(nn.inputLayer.length):
        for k in range(nn.hiddenLayers[0].length):
            pygame.draw.line(screen,pygame.__color_constructor(255-int(255*nn.sigmoid(nn.inputLayer.weights[k][i])),int(255*nn.sigmoid(nn.inputLayer.weights[k][i])),0,255),(LEFT_BORDER,TOP_BORDER + i * round((BOTTOM_BORDER-TOP_BORDER)/nn.inputLayer.length)),
                             (round(LEFT_BORDER + distBetweenLayers),TOP_BORDER + k * round((BOTTOM_BORDER-TOP_BORDER)/nn.hiddenLayers[0].length)),
                             int(5 * nn.sigmoid(nn.inputLayer.weights[k][i]) + 1))

    # drawing weights_2
    for i in range(len(nn.hiddenLayers)-1):
        for j in range(nn.hiddenLayers[i].length):
            for k in range(nn.hiddenLayers[i+1].length):
                pygame.draw.line(screen, pygame.__color_constructor(255 - int(255 * nn.sigmoid(nn.hiddenLayers[i].weights[k][j])),int(255 * nn.sigmoid(nn.hiddenLayers[i].weights[k][j])), 0, 255),
                                 (round(LEFT_BORDER + distBetweenLayers * (i+1)), TOP_BORDER + j * round((BOTTOM_BORDER - TOP_BORDER) / nn.hiddenLayers[i].length)),
                                 (round(LEFT_BORDER + distBetweenLayers * (i+2)), TOP_BORDER + k * round((BOTTOM_BORDER - TOP_BORDER) / nn.hiddenLayers[i+1].length)),
                                 int(5 * nn.sigmoid(nn.hiddenLayers[i].weights[k][j]) + 1))

    # drawing weights_3
    lastHiddenIndex = len(nn.hiddenLayers) - 1
    for i in range(nn.hiddenLayers[lastHiddenIndex].length):
        for k in range(nn.outputLayer.length):
            pygame.draw.line(screen,pygame.__color_constructor(255-int(255*nn.sigmoid(nn.hiddenLayers[lastHiddenIndex].weights[k][i])),int(255*nn.sigmoid(nn.hiddenLayers[lastHiddenIndex].weights[k][i])),0,255),
                             (round(RIGHT_BORDER - distBetweenLayers),TOP_BORDER + i * round((BOTTOM_BORDER-TOP_BORDER)/nn.hiddenLayers[lastHiddenIndex].length)),
                             (RIGHT_BORDER,TOP_BORDER + k * round((BOTTOM_BORDER-TOP_BORDER)/nn.outputLayer.length)),
                             int(5 * nn.sigmoid(nn.hiddenLayers[lastHiddenIndex].weights[k][i]) + 1))

    #drawing inputs
    for k in range(nn.inputLayer.length):
        pygame.draw.circle(screen,pygame.__color_constructor(255 * (1-int(nn.inputLayer.neurons[k])),int(255 * nn.inputLayer.neurons[k]),0,255),
                           (LEFT_BORDER,TOP_BORDER + k * round((BOTTOM_BORDER-TOP_BORDER)/nn.inputLayer.length)),
                           35)
        text = font.render(str(nn.inputLayer.neurons[k]), False, (0, 0, 0))
        screen.blit(text, (LEFT_BORDER,TOP_BORDER + k * round((BOTTOM_BORDER-TOP_BORDER)/nn.inputLayer.length)) )

    #drawing hiddens
    for i in range(len(nn.hiddenLayers)):
        for k in range(nn.hiddenLayers[i].length):
            pygame.draw.circle(screen,pygame.__color_constructor(int(255 * (1-nn.hiddenLayers[i].neurons[k])),int(255 * nn.hiddenLayers[i].neurons[k]),0,255),
                               (round(LEFT_BORDER + distBetweenLayers * (i+1)), TOP_BORDER + k * round((BOTTOM_BORDER-TOP_BORDER)/nn.hiddenLayers[i].length)),
                               int(min(35,(BOTTOM_BORDER - TOP_BORDER)/nn.hiddenLayers[i].length/2)) )
            text = font.render(str(round(nn.hiddenLayers[i].neurons[k],2)), False, (0, 0, 0))
            screen.blit(text, (round(LEFT_BORDER + distBetweenLayers * (i+1)), TOP_BORDER + k * round((BOTTOM_BORDER-TOP_BORDER)/nn.hiddenLayers[i].length)) )

    #drawing outputs
    for k in range(nn.outputLayer.length):
        pygame.draw.circle(screen,pygame.__color_constructor(int(255 * (1-nn.outputLayer.neurons[k])),int(255 * nn.outputLayer.neurons[k]),0,255),
                           (RIGHT_BORDER,TOP_BORDER + k * round((BOTTOM_BORDER-TOP_BORDER)/nn.outputLayer.length)),35)
        text = font.render(str(round(nn.outputLayer.neurons[k],2)), False, (0, 0, 0))
        screen.blit(text, (RIGHT_BORDER,TOP_BORDER + k * round((BOTTOM_BORDER-TOP_BORDER)/nn.outputLayer.length)) )
        text = font.render(str(trainAnswers[x][0]), False, (0, 0, 0))
        screen.blit(text, (RIGHT_BORDER+20,TOP_BORDER + k * round((BOTTOM_BORDER-TOP_BORDER)/nn.outputLayer.length)) )

    pygame.event.get()
    pygame.display.update()

# MAIN VARIABLES
TOP_BORDER = 100
LEFT_BORDER = 100
RIGHT_BORDER = pygame.display.get_surface().get_width() - 100
BOTTOM_BORDER = pygame.display.get_surface().get_height() - 100

maxEpoch = 50000

trainingData = np.array([([0,0,0],[0]),([0,0,1],[1]),([0,1,0],[1]),([0,1,1],[0]),
                     ([1,0,0],[1]),([1,0,1],[0]),([1,1,0],[0]),([1,1,1],[1])])

trainSet = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],
                     [1,0,0],[1,0,1],[1,1,0],[1,1,1]])
trainAnswers = np.array([[0],[1],[1],[0],
                        [1],[0],[0],[1]])

#trainSet = np.array([[0,0],[0,1],[1,0],[1,1]])
#trainAnswers = np.array([[0],[1],[1],[0]])

sumOfCurrentErrors = 0
averageEpochError = 0
errorPoints = []

screen.fill((192, 192, 192))
clock.tick(60)

font = pygame.font.SysFont("comicsansms", 10)

pygame.display.update()


nn = network.Network(len(trainSet[0]), [10], 1)
# nn.load("20180703164533321.txt")

for j in range(0,maxEpoch):
    print("----------- epoch",j,"-----------")
    for i in range(len(trainSet)):
        prediction = nn.feedForward(trainSet[i])
        print("-- set",i,"-- ",prediction,"(",trainAnswers[i],")")
        sumOfCurrentErrors += math.pow(trainAnswers[i] - prediction,2)
        nn.train(trainSet[i],trainAnswers[i])
        drawNet(trainAnswers[i])
    averageEpochError = sumOfCurrentErrors / len(trainSet)
    errorPoints.append(averageEpochError)
    print("averageEpochError ", averageEpochError)
    if averageEpochError < 0.001:
        break
    sumOfCurrentErrors = 0
    averageEpochError = 0
    print()

try:
    X = np.arange(0,maxEpoch,1)

    plt.plot(X,errorPoints)
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    # plt.scatter(X,errorPoints,s=1)
    plt.show()
except Exception:
    print("can not draw plot")

nn.save()