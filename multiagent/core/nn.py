# Tools for building feedforward neural networks and linear approximators
# for use in online reinforcement learning algorithms.

import numpy as np
from scipy.special import expit as sigmoid


# Activation function classes.
class ActivationFunction:
    '''Baseclass for activation functions.'''
    def __init__(self, function, derivative):
        self.f = function
        self.df = derivative
    
    def func(self, x):
        return self.f(x)
    
    def deriv(self, x):
        return self.df(x)
    
    
class SigmoidActivation(ActivationFunction):
    '''Sigmoid activation function.'''
    def __init__(self):
        ActivationFunction.__init__(self, self.sig, self.dsig)
        
    def sig(self, x):
        return sigmoid(x)
    
    def dsig(self, x):
        return sigmoid(x) * (1-sigmoid(x))
    
    
class LinearActivation(ActivationFunction):
    '''Linear activation function.'''
    def __init__(self):
        ActivationFunction.__init__(self, self.x, self.dx)
        
    def x(self, x):
        return x
    
    def dx(self, x):
        return 1


# Linear function approximator with scalar output.
class LinearApproximator:
    def __init__(self, input_dim, initialization_cov_constant=1):
        self.input_dim = input_dim
        self.initialization_cov_constant = initialization_cov_constant
        self.num_betas = self.input_dim + 1
        self.betas = None
        self.reinit_params()
        
    def reinit_params(self):
        self.betas = np.random.multivariate_normal(
                mean=np.zeros(self.num_betas),
                cov=self.initialization_cov_constant*np.eye(self.num_betas))
        
    def get_params(self):
        return self.betas
    
    def set_params(self, new_params):
        self.betas = new_params
        
    def evaluate(self, x):
        x = np.append(x, 1)
        return np.dot(self.betas, x)
    
    def gradient(self, x):
        return np.append(x, 1)

    
# Single layer neural network.
class SingleLayerNN:
    def __init__(self, output_dim, num_hidden_units, input_dim,
                 output_activation, hidden_layer_activation,
                 initialization_cov_constant=1):
        self.output_dim = output_dim
        self.num_hidden_units = num_hidden_units
        self.input_dim = input_dim
        self.out_activation = output_activation
        self.hidden_activation = hidden_layer_activation
        self.initialization_cov_constant = initialization_cov_constant
        
        # The parameters multiplying the input layer are betas,
        # while those multiplying the hidden layer outputs are gammas.
        self.num_gammas = self.output_dim * (self.num_hidden_units+1)
        self.gammas = None
        self.num_betas = self.num_hidden_units * (self.input_dim+1)
        self.betas = None
        self.reinit_params()
        
    def reinit_params(self):
        self.gammas = np.random.multivariate_normal(
                mean=np.zeros(self.num_gammas),
                cov=self.initialization_cov_constant*np.eye(self.num_gammas))
        self.betas = np.random.multivariate_normal(
                mean=np.zeros(self.num_betas),
                cov=self.initialization_cov_constant*np.eye(self.num_betas))
        
    def get_params(self):
        return np.append(self.gammas, self.betas)
    
    def set_params(self, new_params):
        self.gammas = new_params[:self.num_gammas]
        self.betas = new_params[self.num_gammas:]
        
    def _hidden_inputs(self, x):
        x = np.append(x,1)
        beta_matrix = self.betas.reshape(self.num_hidden_units,
                                         self.input_dim+1)
        return np.dot(beta_matrix, x)
        
    def _hidden_outputs(self, x):
        return self.hidden_activation.func(self._hidden_inputs(x))
    
    def _output_inputs(self, hidden):
        hidden = np.append(hidden, 1)
        gamma_matrix = self.gammas.reshape(self.output_dim,
                                           self.num_hidden_units+1)
        return np.dot(gamma_matrix, hidden)
        
    def _output_outputs(self, hidden):
        return self.out_activation.func(self._output_inputs(hidden))
        
    def evaluate(self, x):
        hidden = self._hidden_outputs(x)
        return self._output_outputs(hidden)
    
    def _hidden_derivs(self, x):
        return self.hidden_activation.deriv(self._hidden_inputs(x))
    
    def _output_derivs(self, hidden):
        return self.out_activation.deriv(self._output_inputs(hidden))
        
    def gradient(self, x):
        hidden = self._hidden_outputs(x)
        hidden_derivs = self._hidden_derivs(x)
        output_derivs = self._output_derivs(hidden)
        gammas = self.gammas.reshape(self.output_dim, self.num_hidden_units+1)
        
        f = np.append(hidden, 1)
        G = np.tile(f, (self.output_dim, 1))
        
        x = np.append(x, 1)
        X = np.tile(x, (self.output_dim, 1))
        
        for i in range(self.num_hidden_units):
            newX = hidden_derivs[i] * X * gammas[:,i][:,np.newaxis]
            G = np.concatenate((G, newX), axis=1)
        
        if self.output_dim == 1:
            return output_derivs * G.flatten()
        else:
            return G * output_derivs[:,np.newaxis]
            
        
        
        
        
        
        
# end