import random
import math
import time
import pygame
import numpy as np
import neuralNetwokClass as network
import matplotlib.pyplot as plt

# pygame.init()
# screen = pygame.display.set_mode((600, 400))
# clock = pygame.time.Clock()

# MAIN METHODS
# def drawNet(x):
    # screen.fill((192, 192, 192))

    # # drawing weights_1
    # for i in range(inputNeurons):
    #     for j in range(hiddenNeurons):
    #         pygame.draw.line(screen,pygame.__color_constructor(255-int(255*sigmoid(weights_1[j][i])),int(255*sigmoid(weights_1[j][i])),0,255),(LEFT_BORDER,TOP_BORDER + i * round((BOTTOM_BORDER-TOP_BORDER)/inputNeurons)),
    #                          (round((RIGHT_BORDER - LEFT_BORDER)/2 + LEFT_BORDER),TOP_BORDER +
    #                           j * round((BOTTOM_BORDER-TOP_BORDER)/hiddenNeurons)),
    #                          int(5 * sigmoid(weights_1[j][i]) + 1))
    #
    # # drawing weights_2
    # for i in range(hiddenNeurons):
    #     for j in range(outputNeurons):
    #         pygame.draw.line(screen,pygame.__color_constructor(255-int(255*sigmoid(weights_2[j][i])),int(255*sigmoid(weights_2[j][i])),0,255),(round((RIGHT_BORDER - LEFT_BORDER)/2 + LEFT_BORDER),TOP_BORDER + i * round((BOTTOM_BORDER-TOP_BORDER)/hiddenNeurons)),
    #                          (RIGHT_BORDER, TOP_BORDER + j * round((BOTTOM_BORDER - TOP_BORDER) / outputNeurons)),
    #                          int(5 * sigmoid(weights_2[j][i]) + 1))
    #
    # #drawing inputs
    # for k in range(inputNeurons):
    #     pygame.draw.circle(screen,pygame.__color_constructor(255 * (1-int(inputs[k])),int(255 * inputs[k]),0,255),
    #                        (LEFT_BORDER,TOP_BORDER + k * round((BOTTOM_BORDER-TOP_BORDER)/inputNeurons)),
    #                        35)
    #     text = font.render(str(inputs[k]), False, (0, 0, 0))
    #     screen.blit(text, (LEFT_BORDER,TOP_BORDER + k * round((BOTTOM_BORDER-TOP_BORDER)/inputNeurons)) )
    #
    # #drawing hiddens
    # for k in range(hiddenNeurons):
    #     pygame.draw.circle(screen,pygame.__color_constructor(int(255 * (1-hiddenOutputs[k])),int(255 * hiddenOutputs[k]),0,255),
    #                        (round((RIGHT_BORDER - LEFT_BORDER)/2 + LEFT_BORDER),TOP_BORDER + k * round((BOTTOM_BORDER-TOP_BORDER)/hiddenNeurons)),
    #                        int(min(35,(BOTTOM_BORDER - TOP_BORDER)/hiddenNeurons/2)))
    #     text = font.render(str(round(hiddenOutputs[k],2)), False, (0, 0, 0))
    #     screen.blit(text, (round((RIGHT_BORDER - LEFT_BORDER)/2 + LEFT_BORDER),TOP_BORDER + k * round((BOTTOM_BORDER-TOP_BORDER)/hiddenNeurons)) )
    #
    # #drawing outputs
    # for k in range(outputNeurons):
    #     pygame.draw.circle(screen,pygame.__color_constructor(int(255 * (1-outputOutputs[k])),int(255 * outputOutputs[k]),0,255),
    #                        (RIGHT_BORDER,TOP_BORDER + k * round((BOTTOM_BORDER-TOP_BORDER)/outputNeurons)),35)
    #     text = font.render(str(round(outputOutputs[k],2)), False, (0, 0, 0))
    #     screen.blit(text, (RIGHT_BORDER,TOP_BORDER + k * round((BOTTOM_BORDER-TOP_BORDER)/outputNeurons)) )
    #     text = font.render(str(trainAnswers[x]), False, (0, 0, 0))
    #     screen.blit(text, (RIGHT_BORDER+20,TOP_BORDER + k * round((BOTTOM_BORDER-TOP_BORDER)/outputNeurons)) )
    #
    # pygame.display.update()

# MAIN VARIABLES
# TOP_BORDER = 100
# LEFT_BORDER = 100
# RIGHT_BORDER = pygame.display.get_surface().get_width() - 100
# BOTTOM_BORDER = pygame.display.get_surface().get_height() - 100

maxEpoch = 50000

trainingData = np.array([([0,0,0],[0]),([0,0,1],[1]),([0,1,0],[1]),([0,1,1],[0]),
                     ([1,0,0],[1]),([1,0,1],[0]),([1,1,0],[0]),([1,1,1],[1])])

trainSet = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],
                     [1,0,0],[1,0,1],[1,1,0],[1,1,1]])
trainAnswers = np.array([[0],[1],[1],[0],
                         [1],[0],[0],[1]])

sumOfCurrentErrors = 0
averageEpochError = 0
errorPoints = []

# screen.fill((192, 192, 192))
# clock.tick(60)
#
# font = pygame.font.SysFont("comicsansms", 10)

#pygame.display.update()


nn = network.Network(3, [8,8], 1)
# nn.load("20180703164533321.txt")

for j in range(0,maxEpoch):
    print("----------- epoch",j,"-----------")
    for i in range(len(trainingData)):
        prediction = nn.feedForward(trainSet[i])
        print("-- set",i,"-- ",prediction,"(",trainAnswers[i],")")
        sumOfCurrentErrors += math.pow(trainAnswers[i] - prediction,2)
        nn.train(trainSet[i],trainAnswers[i], 0.001)
    averageEpochError = sumOfCurrentErrors / len(trainSet)
    errorPoints.append(averageEpochError)
    print("averageEpochError ", averageEpochError)
    # if averageEpochError < 0.001:
        # break
    sumOfCurrentErrors = 0
    averageEpochError = 0
    print()


X = np.arange(0,maxEpoch,1)

plt.plot(X,errorPoints)
plt.xlabel("Epochs")
plt.ylabel("MSE")
# plt.scatter(X,errorPoints,s=1)
plt.show()
nn.save()