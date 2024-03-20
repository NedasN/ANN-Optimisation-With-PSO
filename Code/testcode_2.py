import numpy as np
import pandas as pd
import NN
import Particle
import ParticleConversion
import hyperparameter_profiles as profiles
from ParticleSwarmOptimization import ParticleSwarmOptimization

test_profile = profiles.profileArchitecture1

dataset = pd.read_csv("data_banknote_authentication.txt") #Load the dataset
labels = dataset.iloc[:, -1] #store lables from testing data before removing them
dataset.drop(columns=dataset.columns[len(dataset.columns)-1], inplace=True) #Remove labels from testing data

pso = ParticleSwarmOptimization(test_profile,dataset,labels)
pso.pso()

print("PSO Global Best: " + str(pso.global_best))
print("PSO Global Best Position: " + str(pso.global_best_position))
#print("Personal best of first PSO Particle:" + str(pso.particles[0].personal_best))
#print("Position of First Particle: " + str(pso.particles[0].personal_best_position))