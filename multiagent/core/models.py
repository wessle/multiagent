# Policies for use in on/off-policy policy gradient methods.

import numpy as np
from scipy.special import softmax

import multiagent.core.nn as nn


# Value functions:
class ValueFunction:
    '''Empty parent class for specific value functions.'''
    def __init__(self):
        self.v = None
        
    def reinit_params(self):
        self.v.reinit_params()
    
    def get_params(self):
        return self.v.get_params()
    
    def set_params(self, new_params):
        self.v.set_params(new_params)
        
    def evaluate(self, state):
        return self.v.evaluate(state)
    
    def gradient(self, state):
        return self.v.gradient(state)


class ValueFunctionLinear(ValueFunction):
    def __init__(self, state_vector_len, initialization_cov_constant=1):
        
        ValueFunction.__init__(self)
        
        self.v = nn.LinearApproximator(state_vector_len,
                                       initialization_cov_constant)


class ValueFunctionNN(ValueFunction):
    def __init__(self, num_hidden_units, state_vector_len, output_activation,
                 hidden_activation, initialization_cov_constant=1):
        
        ValueFunction.__init__(self)
        
        self.v = nn.SingleLayerNN(1, num_hidden_units, state_vector_len,
                                  output_activation, hidden_activation,
                                  initialization_cov_constant)
    

class NormalizedValueFunctionNN(ValueFunctionNN):
    def __init__(self, state_vector_len, num_hidden_units, output_activation,
                 hidden_activation, initialization_cov_constant=1):
        self.normalizer = 1
        
        ValueFunctionNN.__init__(self, state_vector_len, num_hidden_units,
                               output_activation, hidden_activation,
                               initialization_cov_constant)
        
    def register_normalizer(self, normalizer):
        self.normalizer = normalizer
        
    def _normalstate(self, state):
        return state / self.normalizer
    
    def evaluate(self, state):
        return self.v.evaluate(self._normalstate(state))
    
    def gradient(self, state):
        return self.v.gradient(self._normalstate(state))


# Target policies:
class SoftmaxPolicy:
    '''Empty parent class for use in softmax policies with specific
    h functions.'''
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.h = None
        
    def reinit_params(self):
        self.h.reinit_params()

    def get_params(self):
        return self.h.get_params()
    
    def set_params(self, new_params):
        self.h.set_params(new_params)
    
    def _hvals(self, state):
        return np.array([self.h.evaluate(np.append(state, action))
                         for action in range(self.num_actions)])
    
    def _get_probs(self, state):
        return softmax(self._hvals(state))
        
    def sample_action(self, state):
        probs = self._get_probs(state)
        
        U = np.random.random()
        p = 0
        for i in range(self.num_actions):
            p += probs[i]
            if U < p:
                return i
        return np.random.randint(self.num_actions)
    
    def _hgrads(self, state):
        return np.array([self.h.gradient(np.append(state, action))
                        for action in range(self.num_actions)])
        
    def grad_log_policy(self, state, action):    
        hgrads = self._hgrads(state)
        return hgrads[action] - np.dot(self._get_probs(state).flatten(), hgrads)
    
    def pdf(self, action, state):
        return self._get_probs(state)[action]
    
    
class SoftmaxPolicyLinear(SoftmaxPolicy):
    def __init__(self, feature_vector_size, num_actions,
                 initialization_cov_constant=1):
        
        SoftmaxPolicy.__init__(self, num_actions)
        
        self.h = nn.LinearApproximator(feature_vector_size,
                                       initialization_cov_constant)
    
    
class SoftmaxPolicyNN(SoftmaxPolicy):
    def __init__(self, feature_vector_size, num_actions, num_hidden_units,
                 output_activation, hidden_layer_activation,
                 initialization_cov_constant=1):
        
        SoftmaxPolicy.__init__(self, num_actions)
        
        self.h = nn.SingleLayerNN(1, num_hidden_units, feature_vector_size,
                                  output_activation, hidden_layer_activation,
                                  initialization_cov_constant)


class NormalizedSoftmaxPolicyNN(SoftmaxPolicyNN):
    def __init__(self, feature_vector_size, num_actions, num_hidden_units,
                 output_activation, hidden_layer_activation,
                 initialization_cov_constant=1):
        
        self.action_normalizer = 1
        self.state_normalizer = 1
    
        SoftmaxPolicyNN.__init__(self, feature_vector_size, num_actions,
                               num_hidden_units, output_activation,
                               hidden_layer_activation,
                               initialization_cov_constant)
        
    def register_normalizers(self, state_normalizer, action_normalizer):
        self.state_normalizer = state_normalizer
        self.action_normalizer = action_normalizer
    
    def _normalize(self, x, normalizer):
        return x / normalizer
    
    def _normalstate(self, state):
        return self._normalize(state, self.state_normalizer)
    
    def _normalaction(self, action):
        return self._normalize(action, self.action_normalizer)
    
    def _hvals(self, state):
        state = self._normalstate(state)
        return np.array([self.h.evaluate(
                np.append(state, self._normalaction(action)))
                for action in range(self.num_actions)])

    def _hgrads(self, state):
        state = self._normalstate(state)
        return np.array([self.h.gradient(
                np.append(state, self._normalaction(action)))
                for action in range(self.num_actions)])


# Behavior policies:
class UniformPolicy:
    def __init__(self, num_actions):
        self.num_actions = num_actions
        
    def sample_action(self, state):
        return np.random.randint(self.num_actions)
    
    def pdf(self, action, state):
        return 1/self.num_actions


# end
