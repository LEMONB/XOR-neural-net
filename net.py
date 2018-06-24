import random
import math
import time
import pygame

pygame.init()
screen = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()

def sigmoid(x):
    return 1/(1+math.exp(-x))

def drawNet(x):
    print(inputs[0],inputs[1])
    print(outputOutputs[0],"(",trainAnswers[x],")")
    screen.fill((192, 192, 192))
    pygame.draw.circle(screen,pygame.__color_constructor(255,255,255,255),(100,100),35)
    text = font.render(int.__str__(inputs[0]), False, (0, 0, 0))
    screen.blit(text, (100,100) )
    pygame.draw.circle(screen,pygame.__color_constructor(255,255,255,255),(100,200),35)
    text = font.render(int.__str__(inputs[1]), False, (0, 0, 0))
    screen.blit(text, (100,200) )
    pygame.draw.circle(screen,pygame.__color_constructor(255,255,255,255),(100,300),30)

    text = font.render(float.__str__(round(weightsI2H[0],2)), False, (0, 0, 0))
    screen.blit(text, (200,50) )
    text = font.render(float.__str__(round(weightsI2H[1],2)), False, (0, 0, 0))
    screen.blit(text, (200,100) )

    text = font.render(float.__str__(round(weightsI2H[2],2)), False, (0, 0, 0))
    screen.blit(text, (200,150) )
    text = font.render(float.__str__(round(weightsI2H[3],2)), False, (0, 0, 0))
    screen.blit(text, (200,200) )

    text = font.render(float.__str__(round(weightsI2H[4],2)), False, (0, 0, 0))
    screen.blit(text, (200,250) )
    text = font.render(float.__str__(round(weightsI2H[5],2)), False, (0, 0, 0))
    screen.blit(text, (200,300) )

    text = font.render(float.__str__(round(weightsI2H[6],2)), False, (0, 0, 0))
    screen.blit(text, (200,350) )
    text = font.render(float.__str__(round(weightsI2H[7],2)), False, (0, 0, 0))
    screen.blit(text, (200,400) )
    text = font.render(float.__str__(round(weightsI2H[8],2)), False, (0, 0, 0))
    screen.blit(text, (200,450) )

    pygame.draw.circle(screen,pygame.__color_constructor(255,255,255,255),(300,100),35)
    text = font.render(float.__str__(round(hiddenOutputs[0],2)), False, (0, 0, 0))
    screen.blit(text, (300,100) )
    pygame.draw.circle(screen,pygame.__color_constructor(255,255,255,255),(300,200),35)
    text = font.render(float.__str__(round(hiddenOutputs[1],2)), False, (0, 0, 0))
    screen.blit(text, (300,200) )
    pygame.draw.circle(screen,pygame.__color_constructor(255,255,255,255),(300,300),35)
    text = font.render(float.__str__(round(hiddenOutputs[2],2)), False, (0, 0, 0))
    screen.blit(text, (300,300) )
    pygame.draw.circle(screen,pygame.__color_constructor(255,255,255,255),(300,400),30)

    text = font.render(float.__str__(round(weightsH2O[0],2)), False, (0, 0, 0))
    screen.blit(text, (400,100) )
    text = font.render(float.__str__(round(weightsH2O[1],2)), False, (0, 0, 0))
    screen.blit(text, (400,200) )
    text = font.render(float.__str__(round(weightsH2O[2],2)), False, (0, 0, 0))
    screen.blit(text, (400,300) )
    text = font.render(float.__str__(round(weightsH2O[3],2)), False, (0, 0, 0))
    screen.blit(text, (400,400) )

    pygame.draw.circle(screen,pygame.__color_constructor(255,255,255,255),(500,200),35)
    text = font.render(float.__str__(round(outputOutputs[0],2)), False, (0, 0, 0))
    screen.blit(text, (500, 200) )
    text = font.render(int.__str__(trainAnswers[x]), False, (0, 0, 0))
    screen.blit(text, (550, 200) )
    pygame.display.update()

maxEpoch = 5000
learningSpeed = 0.7
moment = 0.3
bias = 1
trainSet = [[0,1],[1,1],[1,0],[0,0]]
trainAnswers = [1,0,1,0]

randomRange = 0.2
inputs = trainSet[1] #[random.randint(0,1),random.randint(0,1)]
weightsI2H = [random.uniform(-randomRange,randomRange),random.uniform(-randomRange,randomRange),
              random.uniform(-randomRange,randomRange),random.uniform(-randomRange,randomRange),
              random.uniform(-randomRange,randomRange),random.uniform(-randomRange,randomRange),
              random.uniform(-randomRange,randomRange),random.uniform(-randomRange,randomRange),
              random.uniform(-randomRange,randomRange)]
hiddenInputs = []
hiddenOutputs = []
weightsH2O = [random.uniform(-randomRange,randomRange),random.uniform(-randomRange,randomRange),
              random.uniform(-randomRange,randomRange),random.uniform(-randomRange,randomRange)]
outputInputs = []
outputOutputs = []

error = 0
errors = [0,0,0,0]
deltaOut = 0
deltasH = []
lastDeltasH2O = [0,0,0,0]
lastDeltasI2H = [0,0,0,0,0,0,0,0,0]

screen.fill((192, 192, 192))
clock.tick(60)

font = pygame.font.SysFont("comicsansms", 10)

pygame.display.update()

for j in range(0,maxEpoch):
    print("epoch",j)
    for i in range(0,4,1):
        print("set",i)
        inputs = trainSet[i]
        #print("in", inputs[0], inputs[1])

        hiddenInputs = [inputs[0] * weightsI2H[0] + inputs[1] * weightsI2H[3] + bias * weightsI2H[6],
                        inputs[0] * weightsI2H[1] + inputs[1] * weightsI2H[4] + bias * weightsI2H[7],
                        inputs[0] * weightsI2H[2] + inputs[1] * weightsI2H[5] + bias * weightsI2H[8]]
        hiddenOutputs = [sigmoid(hiddenInputs[0]), sigmoid(hiddenInputs[1]), sigmoid(hiddenInputs[2])]

        outputInputs = [hiddenOutputs[0] * weightsH2O[0] + hiddenOutputs[1] * weightsH2O[1] +
                        hiddenOutputs[2] * weightsH2O[2] + bias * weightsH2O[3]]
        outputOutputs = [sigmoid(outputInputs[0])]

        #print("out", outputOutputs)
        errors[i] = math.pow(trainAnswers[i]-outputOutputs[0],2)
        #print("err",i, errors[i])

        # weights from hidden to output
        err = trainAnswers[i] - outputOutputs[0]
        deltaOut = err * ((1 - outputOutputs[0]) * outputOutputs[0])
        deltasH = [(1 - hiddenOutputs[0]) * hiddenOutputs[0] * weightsH2O[0] * deltaOut,
                   (1 - hiddenOutputs[1]) * hiddenOutputs[1] * weightsH2O[1] * deltaOut,
                   (1 - hiddenOutputs[2]) * hiddenOutputs[2] * weightsH2O[2] * deltaOut]
        gradH2O = [hiddenOutputs[0] * deltaOut, hiddenOutputs[1] * deltaOut, hiddenOutputs[2] * deltaOut, bias * deltaOut]

        deltaWeightsH2O = [0] * len(gradH2O)
        for k in range(0,len(gradH2O)):
            deltaWeightsH2O[k] = learningSpeed * gradH2O[k] + moment * lastDeltasH2O[k]

        lastDeltasH2O = deltaWeightsH2O
        for k in range(0, len(weightsH2O)):
            weightsH2O[k] += deltaWeightsH2O[k]

        # weights from input to hidden
        gradI2H = [inputs[0] * deltasH[0], inputs[0] * deltasH[1], inputs[0] * deltasH[2],
                   inputs[1] * deltasH[0], inputs[1] * deltasH[1], inputs[1] * deltasH[2],
                   bias * deltasH[0], bias * deltasH[1], bias * deltasH[2]]

        deltaWeightsI2H = [0] * len(gradI2H)
        for k in range(0,len(gradI2H)):
            deltaWeightsI2H[k] = learningSpeed * gradI2H[k] + moment * lastDeltasI2H[k]

        lastDeltasI2H = deltaWeightsI2H
        for k in range(0, len(weightsI2H)):
            weightsI2H[k] += deltaWeightsI2H[k]

        drawNet(i)
        #time.sleep(0.001)
        print()

    error = sum(errors)/4
    print("err", error)
    print("----------- epoch",j,"------")
    print()

dummy = input()
