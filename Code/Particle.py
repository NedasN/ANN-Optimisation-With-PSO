import numpy as np

#Particle Class which holds a position and a velocity
class Particle:
    #Create a Particle with a position of the vector size, and the velocity being initialized at 0
    def __init__(self,vector_size : int):
        #Initialize the Position
        self.position = np.zeros(vector_size)

        #Initialize the Velocity
        self.velocity = np.zeros(vector_size)

        #Initialize the "Personal Best" - the position at which the Particle had the best performance, and the value of this
        #best performance
        self.personal_best_position = np.zeros(vector_size)
        self.personal_best = 0

        #Set the boundaries for the weights

    
    #Update the Particle's Position based on its Velocity
    def update_position(self):
        self.position = np.add(self.position,self.velocity)

        invalid_values = (self.position < -13) | (self.position > 18)

        # Randomly Reposition particles that are not between -13 and 18
        if np.any(invalid_values):
            self.position[invalid_values] = np.random.rand(np.sum(invalid_values))