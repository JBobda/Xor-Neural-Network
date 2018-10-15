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

#Creating a Neural Network with a 2 input, 3 hidden layer, and 1 output architecture
'''
    2 - 3 - 1
    Neural Net
      --O-
    O-    -
      --O---O
    O-    -
      --O-
'''
brain = nn.NeuralNetwork(2, 3, 1, 0.75)

training_data = []

for i in range(4):
    training_data.append([inputs[i], outputs[i][0]])

for n in range(10000):
    current = random.randint(0, 3)
    brain.train(training_data[current][0], training_data[current][1])


print(brain.predict([1,0]))
print(brain.predict([0,1]))
print(brain.predict([0,0]))
print(brain.predict([1,1]))

