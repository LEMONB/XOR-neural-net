import random
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# MAIN VARIABLES

maxEpoch = 1000

trainingData = np.array([([0, 0, 0], [0]), ([0, 0, 1], [1]), ([0, 1, 0], [1]), ([0, 1, 1], [0]),
                         ([1, 0, 0], [1]), ([1, 0, 1], [0]), ([1, 1, 0], [0]), ([1, 1, 1], [1])])

trainSet = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                     [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
trainAnswers = np.array([[0], [1], [1], [0],
                         [1], [0], [0], [1]])

errorPoints = []

#trainSet = np.array([[0,0],[0,1],[1,0],[1,1]])
#trainAnswers = np.array([[0],[1],[1],[0]])

print("Data: ", trainingData)

model = keras.Sequential()
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(trainSet, trainAnswers,
                    validation_split=0.33, epochs=2000)
print(trainSet)
print(model.predict_proba(trainSet))
# try:
plt.plot(history.history['val_loss'], history.history['loss'])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.ylim(0, 1)
#plt.scatter(np.arange(0,len(errorPoints), dtype=int ),errorPoints, s=10)
plt.show()
# except Exception:
#   print("can not draw plot")
