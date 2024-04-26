class SearchSpace(object):
    def __init__(self):

        self.transformation_space = {
            'skip_connections': ['No', 'Residual', 'Initial', 'Dense', 'Jumping'],
            'dim_hidden': [16, 32, 64, 128, 256],
            'activation_function': ['sigmoid', 'tanh', 'relu', 'linear',
                                    'softplus', 'leaky_relu', 'relu6', 'elu'],
            'num_layers': [1, 2, 4, 6, 8, 10, 12, 14, 16]
        }

        self.propagation_space = {
            'graph_normalization': ['BatchNorm', 'PairNorm', 'NodeNorm', 'MeanNorm'],
            'propagation_layers': [2, 4, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32],
            'aggregator_type': ['sum', 'mean', 'max', 'min']
        }

        self.param_space = {
            'drop_out': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            'learning_rate': [5e-4, 1e-3, 5e-3, 1e-2, 1e-1],
            'weight_decay': [0, 5e-4, 8e-4, 1e-3, 4e-3]
        }

    def get_instance_pso(self, particle_pos):
        transformation_selection = []
        transformation_space = self.transformation_space

        propagation_selection = []
        propagation_space = self.propagation_space

        param_selection = []
        param_space = self.param_space

        k = 0

        for action_name in transformation_space.keys():
            actions = transformation_space[action_name]
            transformation_selection.append(actions[particle_pos[k]])
            k += 1

        for param_name in propagation_space.keys():
            params = propagation_space[param_name]
            propagation_selection.append(params[particle_pos[k]])
            k += 1

        for hyperparam_name in param_space.keys():
            hyperparams = param_space[hyperparam_name]
            param_selection.append(hyperparams[particle_pos[k]])
            k += 1

        return transformation_selection, propagation_selection, param_selection
