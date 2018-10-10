#!/usr/bin/env python3
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

class NeuralNetwork:

    def __init__(self, num_inputs, num_hidden, num_outputs):
        #Takes in the number of input neurons, hidden neurons, and output neurons
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        #Creates Weight and bias matrices based on number of neurons in input, hidden and output
        self.weights0 = np.random.random((self.num_inputs, self.num_hidden))
        self.bias0 = np.random.random((self.num_hidden, 1))
        '''
        print(self.weights0)
        print()
        print(self.bias0)
        print()
        '''
        self.weights1 = np.random.random((self.num_hidden, self.num_outputs))
        self.bias1 = np.random.random((self.num_outputs, 1))
        '''
        print(self.weights1)
        print()
        print(self.bias1)
        '''

    def train(self, inputs):
        print("Training")
    def predict(self, inputs):
        print("Predicting")