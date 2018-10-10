#!/usr/bin/env python3
import neural_network as nn

#XOR Data table
data = [[0, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]]

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
brain = nn.NeuralNetwork(2, 3, 1)
print(brain.predict([[0],
                     [0]]))
