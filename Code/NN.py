import numpy as np

#Neural network class with adjustable number of hidden layers and number of neurons
class NeuralNetwork:
    #The only needed parameter is the layers, an array of numbers, the numbers representing
    #the number of neurons in each layer. The first layer, conceptually, is the input layer, the
    #final layer is the output layer.

    #The other parameter is the activation function for each layer (one less than the number of layers)

    def __init__(self, layer_sizes, activation_functions):
        self.input_vector_size = layer_sizes[0]
        self.output_vector_size = layer_sizes[-1]
        
        if len(activation_functions) != (len(layer_sizes) - 1):
            raise ValueError("The number of activation functions needs to be the number of layers -1!")

        self.layer_sizes = layer_sizes
        self.activation_functions = activation_functions

        # Initialize the weights and biases for the layers

        #1. Initialize random weights for each layer
        self.weights = [np.zeros((self.layer_sizes[i], self.layer_sizes[i+1])) for i in range(len(self.layer_sizes) - 1)]
        #2. Initialize an array of biases of 0 for each layer (except for the output layer)
        self.biases = [np.zeros((1, size)) for size in self.layer_sizes[1:]]

    #Forward Propagation Function
    def forward_propagation(self,input):
        #Initialize the previous output as the input initially, 
        #because that is what is being passed into the first hidden layer
        prev_output = input

        #Get the hidden layers by removing the first and last elements
        hidden_layers = self.layer_sizes[1:]
        hidden_layers = hidden_layers[-1:]

        #Do the matrix calculation over every hidden layer
        for layer in range(len(hidden_layers)):
            #Multiply the weights and inputs(also outputs from previous layers) together
            matrix_product = np.dot(prev_output,self.weights[layer])
            #Add the Biases
            matrix_total = np.add(matrix_product,self.biases[layer])
            #Apply the Activation Function and set the Output of this hidden layer to replace the previous Output
            prev_output = self.activation_functions[layer](matrix_total)
        
        #Now let's deal with the Final Output Layer
        #Same as before, except we use layer+1 (next layer after last hidden layer) 
        # notation to get the final weights listed in self.weights
        #(which will belong to the Output Layer)
        matrix_output_product = np.dot(prev_output, self.weights[layer+1])
        #Add the Biases again using the last indexed biases
        matrix_output_total = np.add(matrix_output_product, self.biases[layer+1])
        #Apply the activation function and set the Output
        output = self.activation_functions[layer+1](matrix_output_total)

        return output