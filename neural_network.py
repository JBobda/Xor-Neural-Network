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
        self.weights0 = np.random.random((self.num_hidden, self.num_inputs))
        self.bias0 = np.random.random((self.num_hidden, 1))
        self.weights1 = np.random.random((self.num_outputs, self.num_hidden))
        self.bias1 = np.random.random((self.num_outputs, 1))

    def train(self, inputs):
        print("Training")

    def predict(self, inputs):
        print("Predicting...")
        #For user conveniance, the input array is transposed into a column vector
        inputs = np.array([inputs])
        inputs = inputs.transpose()
        #Feed forward algortihm, using matrices and weighted sums, we can receive inputs through
        #the input neurons, feed them forward to the hidden layer using dot products, and finally
        #bring them to the output layer with one last dot product, lastly we use the sigmoid 
        #activation function to squish the output between 1 and 0 for all neurons.

        #Calculates the dot product of weights between 1st and 2nd layer and adds bias on top
        hidden_nodes = (np.dot(self.weights0, inputs)) + self.bias0
        #Pipe the hidden nodes through the sigmoid function to squish it between 1 and 0
        hidden_nodes = sigmoid(hidden_nodes)


        #Calculates the dot product of weights between 2nd and 3rd layer and adds bias on top
        output_nodes = np.dot(self.weights1, hidden_nodes) + self.bias1
        #Pipes all of the output through the sigmoid function to squish it between 1 and 0
        output = sigmoid(output_nodes)

        return output