import pandas as pd
import numpy as np
#used to create graphs not sure if we are allowed to leave this in here
#import matplotlib.pyplot as plt 

#load the cognitive and social results
results1 = pd.read_csv("cognitive&social_results.csv")
#load the global results
results2 = pd.read_csv("Global_results.csv")

combined_results = pd.concat([results1,results2],ignore_index=True)

#drop the previous botched indexing
combined_results.drop(columns=combined_results.columns[0],inplace=True)

standard_deviation = pd.DataFrame(columns=['profile_type','weight','average','standard_deviation'])

#we are going to split all the results into the profily_type and the weight, to get the standard deviation for the run
#for every type of profile
for t in combined_results['profile_type'].unique():
    #and every weight tested for that profile
    for weight in combined_results['weight'].unique():
        #make a temporary dataframe to filter the results dataframe
        #get the information for each run of the weight 
        temp_df = combined_results.loc[combined_results['profile_type'] == t]
        temp_df = temp_df.loc[temp_df['weight'] == weight]
        #get the standard deviation of the accuracy the runs 
        sd = temp_df['accuracy'].std()
        #create a standard deviation dataframe with all the information used for calculations removed and made into a simple easy to understand table with as the name implies, standard deviation
        standard_deviation.loc[len(standard_deviation)] = {'profile_type' : t, 'weight' : weight, 'average': temp_df['average'].mean(), 'standard_deviation' : sd}
        

print(standard_deviation)
'''
These loops are for generating graphs, commented out incase we are not allowed to leave matplotlib as an import
for t in standard_deviation['profile_type'].unique():
    tmp_df = standard_deviation.loc[standard_deviation['profile_type'] == t]
    #print(tmp_df)
    tmp_df.plot(x = 'weight', y = 'average')
    plt.title("Avearge accuracy for each weight for profile: " + t)
    plt.xlabel("Weights")
    plt.ylabel("Average Accuracy")
    plt.show()

for t in standard_deviation['profile_type'].unique():
    tmp_df = standard_deviation.loc[standard_deviation['profile_type'] == t]
    #print(tmp_df)
    tmp_df.plot(x = 'weight', y = 'standard_deviation')
    plt.title("Standard Deviation for each weight for profile: " + t)
    plt.xlabel("Weights")
    plt.ylabel("Standard Deviation")
    plt.show()
'''