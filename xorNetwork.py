import random
import math
import time
import pygame
import numpy as np
import datetime
import neuralNetwokClass as network
import nnLayerClass as nl

# pygame.init()
# screen = pygame.display.set_mode((600, 400))
# clock = pygame.time.Clock()

# MAIN METHODS
def sigmoid(x):
    return 1/(1+math.exp(-x))

def saveNN():
    print()
    # timeStamp = str(datetime.datetime.now())
    # fileName = timeStamp
    # fileName = fileName[:-7]
    # fileName += str(inputNeurons) + str(hiddenNeurons) + str(outputNeurons) + ".txt"
    # fileName = fileName.replace(" ","")
    # fileName = fileName.replace("-","")
    # fileName = fileName.replace(":","")
    # file = open(fileName, "w+")
    #
    # for i in range(hiddenNeurons):
    #     for j in range(inputNeurons):
    #         file.write(str(weights_1[i][j])+"\n")
    # file.write("---\n")
    # for i in range(hiddenNeurons):
    #     file.write(str(biases_1[i])+"\n")
    # file.write("---\n")
    #
    # for i in range(outputNeurons):
    #     for j in range(hiddenNeurons):
    #         file.write(str(weights_2[i][j])+"\n")
    # file.write("---\n")
    # for i in range(outputNeurons):
    #     file.write(str(biases_2[i])+"\n")
    # file.write("---\n")
    #
    # file.close()

def loadNN(fileName):
    print()
    # if (str(inputNeurons) + str(hiddenNeurons) + str(outputNeurons)) in fileName:
    #     file = open(fileName,"r")
    #     for i in range(hiddenNeurons):
    #         for j in range(inputNeurons):
    #             weights_1[i][j] = np.float32(file.readline())
    #     file.readline()
    #     for i in range(hiddenNeurons):
    #         biases_1[i] = np.float32(file.readline())
    #     file.readline()
    #
    #     for i in range(outputNeurons):
    #         for j in range(hiddenNeurons):
    #             weights_2[i][j] = np.float32(file.readline())
    #     file.readline()
    #     for i in range(outputNeurons):
    #         biases_2[i] = np.float32(file.readline())
    #     file.readline()
    #
    #     file.close()

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

maxEpoch = 20000

trainSet = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],
                     [1,0,0],[1,0,1],[1,1,0],[1,1,1]])
trainAnswers = np.array([[0],[1],[1],[0],
                         [1],[0],[0],[1]])


averageEpochError = 0

# screen.fill((192, 192, 192))
# clock.tick(60)
#
# font = pygame.font.SysFont("comicsansms", 10)

#pygame.display.update()

#loadNN("20180627005350341.txt")

nn = network.Network(3,[2,2,3],1)

for j in range(0,maxEpoch):
    print("----------- epoch",j,"------")
    for i in range(len(trainSet)):
        print("-- set",i,"--")
        #drawNet(i)
        print(nn.feedForward(trainSet[i]),"(",trainAnswers[i],")")
        nn.train(trainSet[i],trainAnswers[i])
        #time.sleep(0.001)
        print()

    #     for err in errors:
    #         averageEpochError += math.pow(err,2)
    #
    # averageEpochError /= len(trainSet)
    # print("averageEpochError ", averageEpochError)
    # if averageEpochError < 0.001:
    #     break
    #print("----------- epoch",j,"------")
    print()

#saveNN()

dummy = input()