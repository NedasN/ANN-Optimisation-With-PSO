import pandas as pd
from HyperparameterProfile import HyperparameterProfile
from ParticleSwarmOptimization import ParticleSwarmOptimization

import activation_functions as act_func

#Default parameters: a Neural Network with an Input layer of 4, 2 hidden layers of 3 and 2 neurons, and 1 neuron as output, with
#activation functions of ReLU, ReLU, and then Logistic (for an output between 0 and 1, as this is a binary classification problem)
#For PSO, Inertial weight = 0.7, 20 Particles, 200 Iterations, 4 Informants

default_weight_value = 1.25 #The default value for the variables, when unchanged

def create_default_profile() -> HyperparameterProfile:
    new_profile = HyperparameterProfile(
        [4,3,2,1],[act_func.activation_relu,act_func.activation_relu,act_func.activation_logistic],
        0.7,default_weight_value,default_weight_value,default_weight_value,200,20,4
    )
    return new_profile

dataset = pd.read_csv("data_banknote_authentication.txt") #Load the dataset
labels = dataset.iloc[:, -1] #store labels from testing data before removing them
dataset.drop(columns=dataset.columns[len(dataset.columns)-1], inplace=True) #Remove labels from testing data

increments = [0.25,0.5,0.75,1.0,1.25,1.5] #The increments for the different PSO Weights

no_runs = 10 #The number of times to repeat each experiment, to get an average, to compensate for stochasticity

#These are useful to show the progress of the experiment
experiments_to_do = 3 * len(increments) * no_runs #The total number of experiments to do
experiments_done = 0

#use pandas for easier calculations and storage of information
results_df = pd.DataFrame(columns=['profile_type','weight','run','accuracy','average'])


#f = open("experiment_results.txt","w") #Open a file for writing experiment results into

#Carry out experiments with Cognitive Weight - keep social & global weights at default value, 1.25
#f.write("Experimenting with Cognitive Weights:\n")
cognitive_profile = create_default_profile()
for i in increments:
    total = 0.0 #The total score
    cognitive_profile.b = i #Set the profile's cognitive weight to the value in the increment
    temp_df = pd.DataFrame(columns=['profile_type','weight','run','accuracy','average'])
    for x in range(no_runs):
        new_pso = ParticleSwarmOptimization(cognitive_profile,dataset,labels)
        new_pso.pso()
        experiments_done += 1
        temp_df.loc[len(temp_df)] = {'profile_type' : 'cognitive', 'weight' : i, 'run': x, 'accuracy': new_pso.global_best}

        total += new_pso.global_best #Add the global best accuracy of the PSO to the total

        print("Did %d out of %d experiments" % (experiments_done,experiments_to_do))
    
    cognitive_average_accuracy = total / no_runs #Get the average accuracy
    temp_df['average'] = cognitive_average_accuracy
    results_df = pd.concat([results_df,temp_df],ignore_index=True)
    #print(results_df)
    #f.write("\tAverage accuracy for Cognitive Weight of %.2f, after %d runs: %.3f%%\n" % (i,no_runs,cognitive_average_accuracy))

#Carry out experiments with Social Weight - keep cognitive & global weights at default value, 1.25
#f.write("Experimenting with Social Weights:\n")
social_profile = create_default_profile()
for i in increments:
    total = 0.0 #The total score
    social_profile.g = i #Set the profile's social weight to the value in the increment
    temp_df = pd.DataFrame(columns=['profile_type','weight','run','accuracy','average'])
    for x in range(no_runs):
        new_pso = ParticleSwarmOptimization(social_profile,dataset,labels)
        new_pso.pso()
        experiments_done += 1
        temp_df.loc[len(temp_df)] = {'profile_type' : 'Social', 'weight' : i, 'run': x, 'accuracy': new_pso.global_best}

        total += new_pso.global_best #Add the global best accuracy of the PSO to the total

        print("Did %d out of %d experiments" % (experiments_done,experiments_to_do))
    
    social_average_accuracy = total / no_runs #Get the average accuracy
    temp_df['average'] = social_average_accuracy
    results_df = pd.concat([results_df,temp_df],ignore_index=True)
    #f.write("\tAverage accuracy for Social Weight of %.2f, after %d runs: %.3f%%\n" % (i,no_runs,social_average_accuracy))

#can be moved to a different file to run on another machine, since this would have taken 1hour 30mins all together and multithreading is not a part of this project

#f.write("Experimenting with Global Weights:\n")
#Carry out experiments with Global Weight - keep cognitive & social weights at default value, 1.25
global_profile = create_default_profile()
for i in increments:
    total = 0.0 #The total score
    global_profile.gl = i #Set the profile's global weight to the value in the increment
    temp_df = pd.DataFrame(columns=['profile_type','weight','run','accuracy','average'])
    for x in range(no_runs):
        new_pso = ParticleSwarmOptimization(global_profile,dataset,labels)
        new_pso.pso()
        experiments_done += 1
        temp_df.loc[len(temp_df)] = {'profile_type' : 'Global', 'weight' : i, 'run': x, 'accuracy': new_pso.global_best}

        total += new_pso.global_best #Add the global best accuracy of the PSO to the total

        print("Did %d out of %d experiments" % (experiments_done,experiments_to_do))
    
    global_average_accuracy = total / no_runs #Get the average accuracy
    temp_df['average'] = global_average_accuracy
    results_df = pd.concat([results_df,temp_df],ignore_index=True)
    #f.write("\tAverage accuracy for Global Weight of %.2f, after %d runs: %.3f%%\n" % (i,no_runs,global_average_accuracy))

#f.close()
results_df.to_csv('results.csv')