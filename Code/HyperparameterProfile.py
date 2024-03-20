class HyperparameterProfile:
    def __init__(self,network_layer_sizes,network_layer_activations,
                 pso_inertial_weight,pso_cognitive_weight,pso_social_weight,pso_global_weight,
                 pso_iterations,pso_no_particles,pso_no_informants):
        
        #Initialize hyperparameters of the Neural Network
        self.layer_sizes = network_layer_sizes
        self.activation_functions = network_layer_activations
        
        #Initialize the weights
        self.a = pso_inertial_weight
        self.b = pso_cognitive_weight
        self.g = pso_social_weight
        self.gl = pso_global_weight
        
        #Initialize other hyperparameters of PSO
        self.iterations = pso_iterations
        self.no_particles = pso_no_particles

        if pso_no_informants < 1:
            raise ValueError("Need at least 1 informant for PSO.")

        self.no_informants = pso_no_informants
    