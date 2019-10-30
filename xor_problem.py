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
brain = nn.NeuralNetwork(2, 4, 1, 0.25)

#Creating Training data
training_data = []
for i in range(4):
    training_data.append([inputs[i], outputs[i]])


#5000 Epochs of training
for n in range(5000):
    random.shuffle(training_data)
    for i in training_data:
        brain.train(i[0], i[1])


#Calculate the Number of correct answers
correct = 0
incorrect = 0
for i in range(1000):
    pos = random.randint(0, len(training_data)-1)
    prediction = brain.predict(training_data[pos][0])
    target = training_data[pos][1][0]
    rounded = 1 if prediction > 0.7 else 0
    correct = correct + 1 if rounded == target else correct + 0
    incorrect = incorrect + 1 if rounded != target else incorrect + 0


print(str((correct/1000) * 100) + " % Correct" )

#Predictions on the various options
print(brain.predict([1,0]))
print(brain.predict([0,1]))
print(brain.predict([0,0]))
print(brain.predict([1,1]))
