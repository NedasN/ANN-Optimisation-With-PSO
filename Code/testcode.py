import numpy as np
import pandas as pd
import NN
import Particle
import ParticleConversion
import hyperparameter_profiles as profiles
from ParticleSwarmOptimization import ParticleSwarmOptimization

#Choose a profile for testing
test_profile = profiles.profile2

#Testing data and forward propagation
test_data = np.array([[0, 0, 1, 2], [0, 1, 1, 0], [1, 0, 5, 6], [1, 1, 3, 2]])
test_labels = [[0], [1], [1], [0]]
test_labels = pd.DataFrame(test_labels)
test_labels.rename(columns={test_labels.columns[0]: "labels"}, inplace=True)
#actual data
dataset = pd.read_csv("data_banknote_authentication.txt") #Load the dataset
test_size = 0.3 #30% of the dataset is used for testing
test_samples = int(test_size * len(dataset)) #Number of samples used for testing
train_data = dataset.iloc[test_samples:] #Training data
test_d = dataset.iloc[:test_samples] #Testing data

labels = test_d.iloc[:, -1] #store lables from testing data before removing them
test_d.drop(columns=test_d.columns[len(test_d.columns)-1], inplace=True) #Remove labels from testing data
print("labels are:")
print(test_labels)
#print(test_d.head())

neural_network = NN.NeuralNetwork(test_profile.layer_sizes,test_profile.activation_functions)

output = neural_network.forward_propagation(test_data)
print("output is:")
print(output)

#Make sure that we have weights and biases for each layer
assert len(neural_network.weights) == len(neural_network.biases)

#Make sure that, for each element of the test data, there is the corresponding output
assert len(output) == len(test_data)

#Testing particle vector and velocity size correctness
test_particle = Particle.Particle(10)

assert len(test_particle.position) == 10
assert len(test_particle.velocity) == 10

#Testing Neural Network Particle size
neural_network_size = ParticleConversion.get_particle_vector_size(neural_network.layer_sizes)

#The Neural Network base class uses profile2, which is 2 hidden layers of size 4 each.
#The input size is 4
#Therefore, from input to hidden layer 1, there are 4x4 = 16 connections, +4 biases (for the hidden layer)
#Then, from hidden layer 1 to hidden layer 2, there are 4x4 = 16 connections, +4 biases
#Finally, from hidden layer 2 to output, there are 4x1 = 4 connections, +1 bias
#This gives 16 + 4 + 16 + 4 + 4 + 1 = 45
#So, if this Test Neural Network is converted to a Particle, its vector size should be 37
assert neural_network_size == 45

#Furthermore, if this Test Neural Network is converted to a Particle, each layer would have the following sums of
#weights and connections: [20,20,5]
neural_network_layer_sizes = ParticleConversion.get_particle_layer_counts(neural_network.layer_sizes)

assert (neural_network_layer_sizes == [20,20,5]).all()

#Create a 45-size particle - the same size as the Test Neural Network we created
test_particle = Particle.Particle(45)

#Check that its elements are correctly-arranged in the "get_layer_vectors" method
rough_layers = ParticleConversion.get_layer_vectors(neural_network.layer_sizes,test_particle)

#let's check if the fitness function works
s = ParticleConversion.get_particle_vector_size(test_profile.layer_sizes)
#print("s is:")
#print(s)
#print("profilelayer sizes are:")
#print(test_profile.layer_sizes)
new_particle = Particle.Particle(s)
new_particle.position = np.random.rand(s)
print("new particle position is:")
print(new_particle.position)
#Initialize the Particle's Velocity, each dimension's value being between 0 and N/2, where N = no. particles
new_particle.velocity = np.random.rand(s)
print("new particle velocity is:")
print(new_particle.velocity)

#Update the particle's movement
new_particle.update_position()
print("New Particle position after update:")
print(new_particle.position)

#Create a test ParticleSwarmOptimization class
test_pso = ParticleSwarmOptimization(test_profile,test_data,test_labels)

#fitness = test_pso.access_fitness(test_d,labels,test_profile,new_particle)
fitness = test_pso.access_fitness(new_particle)

print("fitness is:")
print(fitness)

assert len(rough_layers[0]) == 20
assert len(rough_layers[1]) == 20
assert len(rough_layers[2]) == 5

#Create a Neural Network based on the layer vectors, and see if it works by checking lengths of weights and biases
particle_neural_network = ParticleConversion.particle_to_neural_network(test_profile.layer_sizes,test_profile.activation_functions,test_particle)

assert len(particle_neural_network.biases[0]) == 4 #Make sure 4 biases from Input to Hidden Layer 1 (H.L. 1)
assert len(particle_neural_network.biases[1]) == 4 #Make sure 4 biases from Hidden Layer 1 to Hidden Layer 2 (H.L. 2)
assert len(particle_neural_network.biases[2]) == 1 #Make sure 1 bias from Hidden Layer 2 to Output

assert len(particle_neural_network.weights[0]) == 4 #Make sure first dimension of weight matrix 2 from Input to H.L. 1
assert len(particle_neural_network.weights[1]) == 4 #Make sure first dimension of weight matrix 4 from H.L. 1 to H.L. 2
assert len(particle_neural_network.weights[2]) == 4 #Make sure first dimension of weight matrix 4 from H.L. 2 to Output

assert len(particle_neural_network.weights[0][0]) == 4 #Make sure second dimension of weight matrix 4 from Input to H.L. 1
assert len(particle_neural_network.weights[1][0]) == 4 #Make sure second dimension of weight matrix 4 from H.L. 1 to H.L. 2
assert len(particle_neural_network.weights[2][0]) == 1 #Make sure second dimension of weight matrix 1 from H.L. 2 to Output

#Run a Test PSO
test_pso.pso()

print("PSO Global Best: " + str(test_pso.global_best))
print("Personal best of first PSO Particle:" + str(test_pso.particles[0].personal_best))
print("Position of First Particle: " + str(test_pso.particles[0].personal_best_position))