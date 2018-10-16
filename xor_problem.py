#!/usr/bin/env python3
import neural_network as nn
import random

#XOR Data table
inputs = [[0, 0],
          [0, 1],
          [1, 0],
          [1, 1]]

outputs = [[0],
           [1],
           [1],
           [0]]

#Creating a Neural Network with a 2 input, 4 hidden layer, and 1 output architecture
#Learning rate of .75
brain = nn.NeuralNetwork(2, 4, 1, 0.75)

#Creating Training data
training_data = []
for i in range(4):
    training_data.append([inputs[i], outputs[i]])


#5000 Epochs of training
for n in range(5000):
    random.shuffle(training_data)
    for i in training_data:
        brain.train(i[0], i[1])


#Predictions on the various options
print(brain.predict([1,0]))
print(brain.predict([0,1]))
print(brain.predict([0,0]))
print(brain.predict([1,1]))
