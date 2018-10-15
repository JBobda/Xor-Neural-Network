#!/usr/bin/env python3
import numpy as np

def sigmoid(x, deriv=False):
    if(deriv):
        return np.multiply(x, 1.0 - x)
    return 1/(1+np.exp(-x))

class NeuralNetwork:

    def __init__(self, num_inputs, num_hidden, num_outputs, learning_rate):
        #Takes in the number of input neurons, hidden neurons, and output neurons
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.learning_rate = learning_rate

        #Creates Weight and bias matrices based on number of neurons in input, hidden and output
        self.weights0 = np.random.random((self.num_hidden, self.num_inputs))
        self.bias0 = np.random.random((self.num_hidden, 1))
        self.weights1 = np.random.random((self.num_outputs, self.num_hidden))
        self.bias1 = np.random.random((self.num_outputs, 1))

    def train(self, inputs, outputs):
        #-------------------------------------------------------------------------------------
        #For user conveniance, the input array is transposed into a column vector
        inputs = np.array([inputs])
        inputs = inputs.transpose()
        hidden_nodes = (np.dot(self.weights0, inputs)) + self.bias0
        #Pipe the hidden nodes through the sigmoid function to squish it between 1 and 0
        hidden_nodes = sigmoid(hidden_nodes)
        #Calculates the dot product of weights between 2nd and 3rd layer and adds bias on top
        output_nodes = np.dot(self.weights1, hidden_nodes) + self.bias1
        #Pipes all of the output through the sigmoid function to squish it between 1 and 0
        predictions = sigmoid(output_nodes)
        #---------------------------------------------------------------------------------------

        #Layer Between Hidden and Output

        #For user conveniance, the outputs array is transposed into a column vector
        outputs = np.array([outputs])
        outputs = outputs.transpose()
        
        #Get Error of output, ERROR = ANSWER - PREDICTION
        output_errors = (outputs - predictions) 

        #Calculate Gradients (Using element-wise multiplication)
        output_gradients = sigmoid(predictions, deriv=True)  
        output_gradients *= output_errors
        output_gradients *= self.learning_rate

        #Calculate hidden -> output Deltas
        hidden_nodes_transpose = hidden_nodes.transpose()
        weights1_deltas = np.dot(output_gradients, hidden_nodes_transpose)
        bias1_deltas = output_gradients

        self.weights1 = self.weights1 + weights1_deltas
        self.bias1 = self.bias1 + bias1_deltas


        #Layer between Input and Hidden

        #Get the transpose of the weight matrix from hidden -> output
        weights1_transpose = self.weights1.transpose()

        #Calculate the error for each of the hidden neurons using matrix multiplication
        hidden_errors = weights1_transpose * output_errors

        #Calculate hidden gradient
        hidden_gradients = sigmoid(hidden_nodes, deriv=True) 
        hidden_gradients *= hidden_errors
        hidden_gradients *= self.learning_rate

        #Calculate input -> hidden Deltas
        inputs_transpose = inputs.transpose()
        weights0_deltas = np.dot(hidden_gradients, inputs_transpose)
        bias0_deltas = hidden_gradients

        self.weights0 = self.weights0 + weights0_deltas
        self.bias0 = self.bias0 + bias0_deltas

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