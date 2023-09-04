
# Simple implementation of a perceptron in Python 

import numpy as np
import math

from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence, rand

# create a random generator based on a seed for reproducibility
# rs = RandomState(MT19937(SeedSequence(123456789)));

import numpy.random as rs

rs.seed(10)

def sigmoid(x):
    return 1 / ( 1 + pow( math.e, -x) )

def sigmoid_derivative(x):
    #sx = sigmoid(x)
    return sigmoid(x) * (1 - sigmoid(x))
    
# a single perceptron
class Neuron :
    
    def __init__(self, n_input = 3, activation_fnc = sigmoid) -> None:
        self.weights = rs.uniform( -1, 1, n_input )  # create as many weights as n_input
        self.f_activation = activation_fnc           # activation function
        self.delta = 0                               # will be used later for backpropagation 
        self.output = 0                              # need for backpropagation
        self.bias  = rs.uniform( -1, 1)
        
        
    def activate( self, input : np.array ):
        
        # S( W * I) + b
        sumweigths = np.sum( np.multiply( self.weights, input ) ) + self.bias 
        
        self.output = self.f_activation( sumweigths )
        return self.output
    
    
class Layer:

    # a layer consists of a set of neurons each having the same number of input
    # and same activation function
    # n_input per neuron
    def __init__(self, n_input = 3, n_neurons = 3, activation_fnc = sigmoid) -> None:
        
        self.neurons = [ Neuron(n_input, activation_fnc ) for _ in range(n_neurons)]
        
        
    def forward( self, input):
        
        return [ neuron.activate(input) for neuron in self.neurons ]
    
    def backward( self, next_layer):

        # for the last layer compute the error and assign to delta
        for li, neuron in enumerate( self.neurons ):
            
            error = sum([n.weights[li] * n.delta for n in next_layer.neurons])
            neuron.delta = error * sigmoid_derivative( neuron.output)

    def update_weights( self, prev_layer, learning_rate):
        
        for neuron in self.neurons:
            
            for j in range(len(neuron.weights)):
                neuron.weights[j] += learning_rate * neuron.delta * prev_layer.neurons[j].output
                
            neuron.bias += learning_rate * neuron.delta  # 4. Apply learning rate when computing the biases

    def update_weights_input( self, input, learning_rate):
        
        for neuron in self.neurons:
            
            for j in range(len(neuron.weights)):
                neuron.weights[j] += learning_rate * neuron.delta * input[j]
                
            neuron.bias += learning_rate * neuron.delta  # 4. Apply learning rate when computing the biases
    
class NeuronalNet :
    
    # an array of layers 
    def __init__(self, layers) -> None:
        
        self.layers = layers
        pass    

    def forward(self, input ) :
        
        crrinput = input
        for layer in self.layers:
            crrinput = layer.forward( crrinput )
            
        return crrinput
    
    def train( self, training_data, labels, epochs, learning_rate):
        
        for epoch in range(epochs):
            for inputs,label in zip( training_data, labels):
                
                # do forward propagation to get result, labe is expected result
                result = self.forward( inputs );
                
                # for the last layer compute the error and assign to delta
                for li, neuron in enumerate( self.layers[-1].neurons):
                    error = label[li]-result[li]
                    neuron.delta = error * sigmoid_derivative( neuron.output)
                       
                if len( self.layers ) > 1:
                    for la in range( len( self.layers) - 2, -1, -1): # go backwards
                        self.layers[la].backward( self.layers[la + 1] )
                        
                self.layers[0].update_weights_input( inputs, learning_rate)
                
                if len( self.layers ) > 1:
                    for la in range( 1, len( self.layers ) ): # go backwards
                        self.layers[la].update_weights( self.layers[la -1 ], learning_rate )
            
                print( "Layer1.delta = ", self.layers[0].neurons[0].delta, "; bias = ", self.layers[0].neurons[0].bias, "; outputs[1] = ", self.layers[0].neurons[0].output, "; weights[1] = ", self.layers[0].neurons[0].weights )    	
                print( "Layer2.delta = ", self.layers[1].neurons[0].delta, "; bias = ", self.layers[1].neurons[0].bias, "; outputs[1] = ", self.layers[1].neurons[0].output, "; weights[1] = ", self.layers[1].neurons[0].weights )    	
                print( "Layer3.delta = ", self.layers[2].neurons[0].delta, "; bias = ", self.layers[2].neurons[0].bias, "; outputs[1] = ", self.layers[2].neurons[0].output, "; weights[1] = ", self.layers[2].neurons[0].weights )    	
                print( "********************")            
                    
                        

        
    
# -------------- test the code --------------

print( "Sigmoid(0) = ", sigmoid(0) ); # test sigmoid

# # create a neuron
# perceptron = Neuron( 4 )
# result = perceptron.activate( [1,1,1,1] )
# print( "Perceptron.weights = ", perceptron.weights )
# print( "Perceptron.activate( [1,1,1,1] ) = ", result )

# layer = Layer( 4, 5 )

# print( "Layer.neurons.length = ", len( layer.neurons ) )
# print( "Layer.neurons.weights = ", layer.neurons[0].weights )

# print( "Layer.forward( [1,1,1,1] ) = ", layer.forward( [1,1,1,1] ) )



# # Test another neuron
# neuron = Neuron(3)  # Create a neuron with 3 inputs
# neuron_inputs = [0.5, 0.2, -0.8]  # Generate inputs for the neuron
# neuron_output = neuron.activate(neuron_inputs)  # Pass the inputs to the created neuron

# print(f'Output of the neuron is {neuron_output:.3f}')


# # lets create a neuronal net with three layers
# # 2 inputs for first layer
# # three neurons per hidden layer

# layer1 = Layer( n_input = 2, n_neurons = 3 ) # 2 inputs for the model and 3 neurons 
# layer2 = Layer( n_input = 3, n_neurons = 3 ) # need 3 inputs for each output from previous layer
# layer3 = Layer( n_input = 3, n_neurons = 1 ) # need 3 inputs for each output from previous layer and produce 1 output = 1 neuron

# layers = [layer1, layer2, layer3]
# nn = NeuronalNet(layers=layers)

# # Show weights of the third neuron in the second hidden layer
# print([round(weight, 2) for weight in nn.layers[1].neurons[2].weights])

# # Show weights of the single neuron in the output layer
# print([round(weight, 2) for weight in nn.layers[2].neurons[0].weights])

# print( "nn.forward( [0, 1] ) = ", nn.forward( [0,1] ))

    
# # Try X-Or problem    
# xor_0_0 = round(nn.forward([0, 0])[0], 3) 
# xor_0_1 = round(nn.forward([0, 1])[0], 3)
# xor_1_0 = round(nn.forward([1, 0])[0], 3)
# xor_1_1 = round(nn.forward([1, 1])[0], 3)

# # Print ouptut of the perceptron and its rounded value
# print('0 xor 0 =', xor_0_0, '=', round(xor_0_0))
# print('0 xor 1 =', xor_0_1, '=', round(xor_0_1))
# print('1 xor 0 =', xor_1_0, '=', round(xor_1_0))
# print('1 xor 1 =', xor_1_1, '=', round(xor_1_1))


layer1 = Layer( n_input = 2, n_neurons = 6 ) # 2 inputs for the model and 3 neurons 
layer2 = Layer( n_input = 6, n_neurons = 6 ) # need 3 inputs for each output from previous layer
layer3 = Layer( n_input = 6, n_neurons = 1 ) # need 3 inputs for each output from previous layer and produce 1 output = 1 neuron

layers = [layer1, layer2, layer3]
model = NeuronalNet(layers=layers)


data = [[0, 0], [0, 1], [1, 0], [1, 1]]  # Inputs for the XOR problem
labels = [[0], [1], [1], [0]]  # Target labels for the XOR problem
model.train(data, labels, 3, 0.2)  # Training the perceptron

# Test the trained perceptron on the XOR problem.
for d in data:
    print(f'{d} => {round(model.forward(d)[0], 3)} => {round(model.forward(d)[0])}')