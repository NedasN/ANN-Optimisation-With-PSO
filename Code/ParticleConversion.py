import numpy as np
from NN import NeuralNetwork
from Particle import Particle

#For concept's sake, here is an explanation of how a Neural Network would be codified as a particle:
#1. Go through every layer (except for the input layer)
#2. Get the weight matrix from the previous layer to this layer
#3. Flatten the weight matrix into a vector
#4. Concatenate the flattened weight vector with the biases for this layer
#5. All of the vectors for each layer concatenated together represent a particle's position

#There is no reason to write code to convert a Neural Network into a Particle, since the Neural Networks themselves
#Will only be evaluated and will do feeds forward - they will not be altered in any way, which means converting them
#into particles is unnecessary.

#Code that returns a list of numbers. Each number in this list represents the sum of the number of weights and biases
#for that layer in the Neural Network - this will be useful in converting Particles to Neural Networks
def get_particle_layer_counts(layer_sizes) -> np.array:
    #Initialize the empty array
    result = []

    for layer in range(len(layer_sizes)):
        #If layer is 0, it is the first layer, so ignore it
        if layer == 0:
            continue
        else:
            #Otherwise, get the size of the previous layer and multiply it by the size of this layer,
            #which gives the size of the Weight Matrix (since every neuron is connected to every other neuron)
            weight_size = layer_sizes[layer] * layer_sizes[layer-1]

            #The number of biases is simply the number of neurons in this layer
            bias_size = layer_sizes[layer]

            #Add the number of weights and biases to the total result
            result.append(weight_size + bias_size)
    
    return np.array(result)

#Code for getting the size of a Particle's Vector if a Neural Network is converted into a Particle
#This is for information reasons
def get_particle_vector_size(layer_sizes) -> int:
    return np.sum(get_particle_layer_counts(layer_sizes))

#Code for converting a Particle's Vector, a long list of numbers, into a list of smaller lists of numbers,
#each smaller list representing all the weights and biases for each layer of the resulting Neural Network from this particle
#As described in the concept at the top of this script, this list is a list of "Layer Vectors", which are concatenations of
#The Flattened Weight Matrix and Biases Vector for each layer.
def get_layer_vectors(layer_sizes, particle : Particle):
    #Get the number of elements in each layer
    layer_counts = get_particle_layer_counts(layer_sizes)

    resulting_layers = []

    #Copy the particle's position, so that we can pop elements from it easily
    removable_array = particle.position.copy()

    #Go through every layer in the layer count, adding elements corresponding to its
    #index to the resulting layers
    for layer in range(len(layer_counts)):
        new_rough_layer = []

        for element in range(layer_counts[layer]):
            #Add the first element of the removable array to this layer, then
            #remove it from the list
            new_rough_layer.append(removable_array[0])
            removable_array = removable_array[1:]
        
        resulting_layers.append(new_rough_layer)
    
    return resulting_layers

#Code to convert a Particle into a Neural Network
def particle_to_neural_network(layer_sizes, activation_functions, particle : Particle):
    #Initialize the Neural Network corresponding to this Particle
    neural_network = NeuralNetwork(layer_sizes, activation_functions)

    #Initialize empty array for the NN's biases
    particle_biases = []
    
    #Initialize empty array for the NN's Weight Matrices
    particle_weights = []

    #First, get layer vectors, i.e. the biases and flattened weight matrices for each layer, arranged into that layer
    layer_vectors = get_layer_vectors(layer_sizes, particle)

    #Next, go through each rough layer, extracting the biases
    #As described in the concept at the top of this file, biases are always at the end of each layer's vector.
    #Since we have extracted the layer vectors, per the concept described at the top of this script, we can
    #Just use an array slicing operation to "skim off" the biases from the layer vectors, leaving us with just
    #Flattened matrices
    for l in range(len(layer_vectors)):
        layer = l+1 #Add 1 to the layer because we ignore the input layer
        #I.e. index 0 will be layer 1, i.e. the layer vector of weights and biases between the Input layer
        #and the first hidden layer

        bias_count = layer_sizes[layer] #Get the number of biases for the layer, which is just no. neurons in that layer

        #Extract the biases - get the last N elements of the layer vector, where N is the size of this layer 
        # (i.e. no. neurons in this layer)

        layer_biases = layer_vectors[l][-bias_count:]

        #Add the biases for this layer to the particle biases
        particle_biases.append(np.array(layer_biases))

        #Remove the biases from the layer vector, so that we can focus on just the flattened matrix
        del(layer_vectors[l][-bias_count:])
    
    #With the Biases extracted, we simply set the Neural Network's Biases to the ones we extracted
    neural_network.biases = particle_biases

    #We are now left with smaller layer vectors, and each layer vector now represents a flattened weight matrix
    #So, we go through them again, get the dimension of the matrix (which we can infer from size of this layer and previous layer),
    #and use a Numpy function to convert this array into a Matrix
    for l in range(len(layer_vectors)):
        layer = l+1 #Add 1 to the layer because we ignore input layer
        #I.e. index 0 will be the layer vector of flattened weights between the Input layer and first hidden layer

        matrix_m = layer_sizes[layer-1] #Get the size of the previous layer - this is the 1st dimension of the Matrix
        matrix_n = layer_sizes[layer] #Get the size of this layer - this is the 2nd dimension of the Matrix

        weight_matrix = np.reshape(layer_vectors[l], (matrix_m,matrix_n))

        particle_weights.append(np.array(weight_matrix))
    
    #With the weights extracted, set the Neural Network's Weights to the ones we acquired
    neural_network.weights = particle_weights

    #Finally, return the Neural Network!

    return neural_network