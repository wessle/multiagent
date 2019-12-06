# Agents.
#
# TODO: reduce the amount of repetition in this file.
# TODO: get rid of normalization business or make it work.

import numpy as np

import multiagent.core.nn as nn
import multiagent.core.models as models


MIN_RHO_VAL = 0.001


class SARSAAgent:
    '''Basic SARSA agent.'''
    def __init__(self, alpha, epsilon, gamma,
                 output_dim, hidden_units, input_dim, num_actions):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.curr_state = np.array([None, None])
        self.curr_action = 0
        self.next_action = 0
        self.q = nn.SingleLayerNN(output_dim, hidden_units, input_dim,
                                  nn.LinearActivation(),
                                  nn.SigmoidActivation())
        self.num_actions = num_actions

    def sample_action(self, state):
        self.curr_state = state
        possible_values = np.array(
                [self.q.evaluate(np.append(state, a)) \
                 for a in range(self.num_actions)])
        greedy = np.argmax(possible_values)
        
        random = np.random.uniform()
        if random > self.epsilon:
            self.next_action = greedy
        else:
            self.next_action = np.random.randint(self.num_actions)
            
        return self.next_action

    def update(self, next_state, reward):
        curr_feature = np.append(self.curr_state, self.curr_action)
        next_feature = np.append(next_state, self.next_action)

        curr_val = self.q.evaluate(curr_feature)
        next_val = self.q.evaluate(next_feature)
        grad = self.q.gradient(curr_feature)
        self.q.set_params(self.q.get_params() +
            self.alpha*(reward + self.gamma*next_val - curr_val)*grad)
        
        self.curr_action = self.next_action
        

class QLearningAgent:
    '''Basic Q-learning agent.'''
    def __init__(self, alpha, epsilon, gamma, output_dim, hidden_units,
                 input_dim, num_actions):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.curr_state = np.array([None, None])
        self.curr_action = 0
        self.q = nn.SingleLayerNN(output_dim, hidden_units, input_dim,
                                  nn.LinearActivation(),
                                  nn.SigmoidActivation())
        self.num_actions = num_actions
        
    def _best_action(self, state):
        possible_values = np.array(
                [self.q.evaluate(np.append(state, a)) \
                 for a in range(self.num_actions)])
        greedy = np.argmax(possible_values)
        return greedy
        
    def sample_action(self, state):
        self.curr_state = state
        greedy = self._best_action(state)
        
        random = np.random.uniform()
        if random > self.epsilon:
            self.next_action = greedy
        else:
            self.next_action = np.random.randint(self.num_actions)
            
        return self.next_action
    
    def update(self, next_state, reward):
        curr_feature = np.append(self.curr_state, self.curr_action)
        curr_val = self.q.evaluate(curr_feature)
        grad = self.q.gradient(curr_feature)
        next_feature = np.append(next_state, self._best_action(next_state))
        next_val = self.q.evaluate(next_feature)
        new_params = self.q.get_params() + self.alpha * (
                reward + self.gamma * next_val - curr_val) * grad
        self.q.set_params(new_params)


class ACAgent:
    '''Basic actor-critic agent with eligibility traces.'''
    def __init__(self, policy, value_function, actor_stepsize, critic_stepsize,
                 lambda_pi, lambda_v, gamma, clip_grads=False, agent_id=0):
        self.pi = policy
        self.v = value_function
        self.actor_stepsize = actor_stepsize
        self.critic_stepsize = critic_stepsize
        self.lambda_pi = lambda_pi
        self.lambda_v = lambda_v
        self.gamma = gamma
        self.clip_grads = clip_grads
        self.agent_id = agent_id
        
        self.z_pi = 0*self.pi.get_params()    # eligibility trace for pi
        self.z_v = 0*self.v.get_params()    # eligibility trace for v
        self.curr_state = None
        self.curr_action = None
        
    def reinit_params(self):
        self.pi.reinit_params()
        self.v.reinit_params()
        
    def sample_action(self, state):
        self.curr_state = state
        self.curr_action = self.pi.sample_action(state)
        return self.curr_action
    
    def _clipper(self, x):
        if self.clip_grads:
            return np.clip(x, -1, 1)
        else:
            return x
    
    def update(self, next_state, reward):
        delta = reward + self.gamma * self.v.evaluate(next_state) \
            - self.v.evaluate(self.curr_state)
        
        # Critic update.
        self.z_v = self.gamma * self.lambda_v * self.z_v + \
            self._clipper(self.v.gradient(self.curr_state))
        new_v_params = self.v.get_params() + self.critic_stepsize * delta * self.z_v
        self.v.set_params(new_v_params)
        
        # Actor update.
        self.z_pi = self.gamma * self.lambda_pi * self.z_pi + \
            self._clipper(self.pi.grad_log_policy(self.curr_state,
                                                  self.curr_action))
        new_pi_params = self.pi.get_params() + \
            self.actor_stepsize * delta * self.z_pi
        self.pi.set_params(new_pi_params)
            
        # Transition to next state.
        self.curr_state = next_state

    def change_stepsizes(self, new_actor_stepsize, new_critic_stepsize):
        self.actor_stepsize = new_actor_stepsize
        self.critic_stepsize = new_critic_stepsize
    
    
class LinearACAgent(ACAgent):
    def __init__(self, state_vector_len, action_vector_len, num_actions,
                 actor_stepsize, critic_stepsize, lambda_pi, lambda_v,
                 gamma, policy_cov_constant=1, value_func_cov_constant=1,
                 clip_grads=False):
        
        value_func = models.ValueFunctionLinear(state_vector_len,
                                                value_func_cov_constant)
        
        policy = models.SoftmaxPolicyLinear(
                state_vector_len + action_vector_len, num_actions,
                policy_cov_constant)
        
        ACAgent.__init__(self, policy, value_func,
                         actor_stepsize, critic_stepsize,
                         lambda_pi, lambda_v, gamma, clip_grads)
   

class NeuralACAgent(ACAgent):
    def __init__(self, num_policy_hidden_units, num_value_func_hidden_units,
                 policy_output_activation, policy_hidden_activation,
                 value_func_output_activation, value_func_hidden_activation,
                 state_vector_len, action_vector_len, num_actions,
                 actor_stepsize, critic_stepsize, lambda_pi, lambda_v, gamma,
                 policy_cov_constant=1, value_func_cov_constant=1,
                 clip_grads=False, normalize=False):
        
        self.normalize = normalize
        
        if normalize:
            value_func = models.NormalizedValueFunctionNN(
                    num_value_func_hidden_units, state_vector_len,
                    value_func_output_activation, value_func_hidden_activation,
                    value_func_cov_constant)
            
            policy = models.NormalizedSoftmaxPolicyNN(
                    state_vector_len + action_vector_len, num_actions,
                    num_policy_hidden_units, policy_output_activation,
                    policy_hidden_activation, policy_cov_constant)
            
        else:
            value_func = models.ValueFunctionNN(
                    num_value_func_hidden_units, state_vector_len,
                    value_func_output_activation, value_func_hidden_activation,
                    value_func_cov_constant)
            
            policy = models.SoftmaxPolicyNN(
                    state_vector_len + action_vector_len, num_actions,
                    num_policy_hidden_units, policy_output_activation,
                    policy_hidden_activation, policy_cov_constant)
        
        ACAgent.__init__(self, policy, value_func, actor_stepsize,
                         critic_stepsize, lambda_pi, lambda_v, gamma,
                         clip_grads)
        
    def register_normalizers(self, state_normalizer, action_normalizer):
        if self.normalize:
            self.v.register_normalizer(state_normalizer)
            self.pi.register_normalizers(state_normalizer,
                                         action_normalizer)
        else:
            pass
        
    
class SimpleNeuralACAgent(NeuralACAgent):
    '''This simple agent assumes that actions are integers.'''
    def __init__(self, policy_units, value_units, state_vector_len,
                 num_actions, actor_stepsize, critic_stepsize,
                 lambda_pi, lambda_v, gamma, cov=1, clip_grads=False,
                 normalize=False):
        NeuralACAgent.__init__(self, policy_units, value_units,
                               nn.LinearActivation(), nn.SigmoidActivation(),
                               nn.LinearActivation(), nn.SigmoidActivation(),
                               state_vector_len, 1, num_actions,
                               actor_stepsize, critic_stepsize,
                               lambda_pi, lambda_v, gamma, cov, cov,
                               clip_grads, normalize)
    

class OPACAgent(ACAgent):
    '''Off-policy actor-critic agent.'''
    
    # TODO: add clip_rho=True and make appropriate changes.
    def __init__(self, behavior_policy, target_policy, value_function,
                 actor_stepsize, critic_stepsize, lambda_pi, lambda_v, gamma,
                 clip_grads=False, clip_rho=True):        
        
        ACAgent.__init__(self, target_policy, value_function, actor_stepsize,
                         critic_stepsize, lambda_pi, lambda_v, gamma,
                         clip_grads)
        
        self.mu = behavior_policy
        self.clip_rho = clip_rho
        self.prev_F = 0
        self.prev_rho = 1
        self.curr_rho = 1
        
    def update_rho(self):
        pi_t = self.pi.pdf(self.curr_action, self.curr_state)
        mu_t = self.mu.pdf(self.curr_action, self.curr_state)
        rho_t = max(MIN_RHO_VAL, pi_t / mu_t)
        
        # Clip the importance sampling ratio to avoid blowing up.
        # This is not completely kosher theory-wise, but can be explained
        # by relying on Retrace/V-trace-type arguments.
        if self.clip_rho:
            self.curr_rho = min(1, rho_t)
        else:
            self.curr_rho = rho_t
            
    def set_rho(self, rho):
        self.curr_rho = rho
        
    def get_rho(self):
        return self.curr_rho
        
    def sample_behavior_action(self, state):
        self.curr_state = state
        self.curr_action = self.mu.sample_action(state)
        return self.curr_action
    
    def sample_target_action(self, state):
        return self.pi.sample_action(state)
    
    def _clipper(self, x):
        if self.clip_grads:
            return np.clip(x, -1, 1)
        else:
            return x
        
    def update(self, next_state, reward):
        curr_rho = self.curr_rho
        
        curr_F = 1 + self.gamma * self.prev_rho * self.prev_F
        delta = reward + self.gamma * self.v.evaluate(next_state) \
            - self.v.evaluate(self.curr_state)
        M_v = self.lambda_v + (1 - self.lambda_v) * curr_F
        M_pi = 1 + self.lambda_pi * self.gamma * self.prev_rho * self.prev_F
        
        # critic trace update
        self.z_v = self.gamma * self.lambda_v * self.z_v \
            + M_v * self._clipper(self.v.gradient(self.curr_state))
        
        # critic update
        new_v_params = self.v.get_params() \
            + self.critic_stepsize * curr_rho * delta * self.z_v
        self.v.set_params(new_v_params)
        
        # actor update
        grad_log_pi = self._clipper(
                self.pi.grad_log_policy(self.curr_state, self.curr_action))
        new_pi_params = self.pi.get_params() \
            + self.actor_stepsize * curr_rho * M_pi * grad_log_pi * delta
        self.pi.set_params(new_pi_params)
        
        self.prev_F = curr_F
        self.prev_rho = curr_rho
        
        self.update_rho()
        
        # Transition to next state
        # TODO: is this really necessary, given sample_behavior_action?
        self.curr_state = next_state
    
    
class LinearOPACAgent(OPACAgent):
    def __init__(self, behavior_policy,
                 state_vector_len, action_vector_len, num_actions,
                 actor_stepsize, critic_stepsize, lambda_pi, lambda_v, gamma,
                 policy_cov_constant=1, value_func_cov_constant=1,
                 clip_grads=False, clip_rho=True):
        
        value_func = models.ValueFunctionLinear(state_vector_len,
                                                value_func_cov_constant)
        
        target_policy = models.SoftmaxPolicyLinear(
                state_vector_len + action_vector_len, num_actions,
                policy_cov_constant)
        
        OPACAgent.__init__(self, behavior_policy, target_policy,
                           value_func, actor_stepsize, critic_stepsize,
                           lambda_pi, lambda_v, gamma, clip_grads, clip_rho)
        

class SimpleLinearOPACAgent(LinearOPACAgent):
    '''Assume the actions are integers and use a uniform behavior policy.'''
    def __init__(self, state_vector_len, num_actions, actor_stepsize,
                 critic_stepsize, lambda_pi, lambda_v, gamma, cov=1,
                 clip_grads=False, clip_rho=True):
        
        action_vector_len = 1
        
        LinearOPACAgent.__init__(self, models.UniformPolicy(num_actions),
                                 state_vector_len, action_vector_len,
                                 num_actions, actor_stepsize, critic_stepsize,
                                 lambda_pi, lambda_v, gamma, cov, cov,
                                 clip_grads, clip_rho)
        
class NNPolicyLinearOPACAgent(OPACAgent):
    '''Use a neural network policy and linear value function approximation.'''
    def __init__(self, num_target_policy_hidden_units,
                 target_policy_output_activation,
                 target_policy_hidden_activation,
                 state_vector_len, action_vector_len, num_actions,
                 actor_stepsize, critic_stepsize,
                 lambda_pi, lambda_v, gamma,
                 value_func_cov_constant=1.0,
                 policy_cov_constant=1.0,
                 clip_grads=False, clip_rho=True):
        
        value_func = models.ValueFunctionLinear(state_vector_len,
                                                value_func_cov_constant)
        
        target_policy = models.SoftmaxPolicyNN(
                state_vector_len + action_vector_len, num_actions,
                num_target_policy_hidden_units,
                target_policy_output_activation,
                target_policy_hidden_activation, policy_cov_constant)
        
        behavior_policy = models.UniformPolicy(num_actions)
        
        OPACAgent.__init__(self, behavior_policy, target_policy,
                           value_func, actor_stepsize, critic_stepsize,
                           lambda_pi, lambda_v, gamma, clip_grads, clip_rho)
        

class SimpleNNPolicyLinearOPACAgent(NNPolicyLinearOPACAgent):
    '''Use standard activations, assume actions are integers.'''
    def __init__(self, num_target_policy_hidden_units,
                 state_vector_len, num_actions,
                 actor_stepsize, critic_stepsize,
                 lambda_pi, lambda_v, gamma, cov=1.0,
                 clip_grads=False, clip_rho=True):
        
        action_vector_len = 1
        
        NNPolicyLinearOPACAgent.__init__(
                self, num_target_policy_hidden_units,
                nn.LinearActivation(), nn.SigmoidActivation(),
                state_vector_len, action_vector_len,
                num_actions, actor_stepsize, critic_stepsize,
                lambda_pi, lambda_v, gamma, cov,
                clip_grads, clip_rho)
    
    
class NeuralOPACAgent(OPACAgent):
    def __init__(self, behavior_policy,
                 num_target_policy_hidden_units,
                 num_value_func_hidden_units,
                 target_policy_output_activation,
                 target_policy_hidden_activation,
                 value_func_output_activation,
                 value_func_hidden_activation,
                 state_vector_len, action_vector_len, num_actions,
                 actor_stepsize, critic_stepsize, lambda_pi, lambda_v,
                 gamma, policy_cov_constant=1, value_func_cov_constant=1,
                 clip_grads=False, normalize=False):
        
        self.normalize = normalize
        
        if normalize:
            target_policy = models.NormalizedSoftmaxPolicyNN(
                    state_vector_len + action_vector_len, num_actions,
                    num_target_policy_hidden_units,
                    target_policy_output_activation,
                    target_policy_hidden_activation, policy_cov_constant)
    
            value_function = models.NormalizedValueFunctionNN(
                    num_value_func_hidden_units, state_vector_len,
                    value_func_output_activation, value_func_hidden_activation,
                    value_func_cov_constant)
        else:
            target_policy = models.SoftmaxPolicyNN(
                    state_vector_len + action_vector_len, num_actions,
                    num_target_policy_hidden_units,
                    target_policy_output_activation,
                    target_policy_hidden_activation, policy_cov_constant)
    
            value_function = models.ValueFunctionNN(
                    num_value_func_hidden_units, state_vector_len,
                    value_func_output_activation, value_func_hidden_activation,
                    value_func_cov_constant)
        
        OPACAgent.__init__(self, behavior_policy, target_policy,
                           value_function, actor_stepsize, critic_stepsize,
                           lambda_pi, lambda_v, gamma, clip_grads)
        
    def register_normalizers(self, state_normalizer, action_normalizer):
        if self.normalize:
            self.v.register_normalizer(state_normalizer)
            self.pi.register_normalizers(state_normalizer,
                                         action_normalizer)
        else:
            pass
        

class SimpleNeuralOPACAgent(NeuralOPACAgent):
    '''Assume the actions are integers, use a uniform behavior policy.'''
    def __init__(self, policy_units, value_units, state_vector_len,
                 num_actions, actor_stepsize, critic_stepsize,
                 lambda_pi, lambda_v, gamma, cov=1, clip_grads=False,
                 normalize=False):
        
        action_vector_len = 1
        
        NeuralOPACAgent.__init__(self, models.UniformPolicy(num_actions),
                                 policy_units, value_units,
                                 nn.LinearActivation(), nn.SigmoidActivation(),
                                 nn.LinearActivation(), nn.SigmoidActivation(),
                                 state_vector_len, action_vector_len,
                                 num_actions, actor_stepsize, critic_stepsize,
                                 lambda_pi, lambda_v, gamma, cov, cov,
                                 clip_grads, normalize)

        
class MetaACAgent:
    '''Collection of multiple ACAgents for use in multi-agent environments.'''
    def __init__(self, agent, num_agents):
        self.num_agents = num_agents
        self.agents = [agent for i in range(self.num_agents)]
        for i in range(1, self.num_agents):
            self.agents[i].agent_id = i
        self.reinit_agent_params()
            
    def reinit_agent_params(self):
        for agent in self.agents:
            agent.reinit_params()
            
    def get_value_params(self):
        return np.array([agent.v.get_params() for agent in self.agents])
    
    def set_value_params(self, new_params):
        for i in range(self.num_agents):
            self.agents[i].v.set_params(new_params[i])
        
    def sample_action(self, state):
        return np.array([agent.sample_action(state) for agent in self.agents])

    def update(self, next_state, rewards):
        for i in range(self.num_agents):
            self.agents[i].update(next_state, rewards[i])
            
    def change_stepsizes(self, new_actor_stepsize, new_critic_stepsize):
        for agent in self.agents:
            agent.change_stepsizes(new_actor_stepsize, new_critic_stepsize)
            
    def get_average_state_value(self, state):
        return np.average([agent.v.evaluate(state) for agent in self.agents])
            
            
class MetaOPACAgent(MetaACAgent):
    '''Collection of multiple OPACAgents for use in multi-agent environments.'''
    def __init__(self, agent, num_agents):
        MetaACAgent.__init__(self, agent, num_agents)
        
    def sample_behavior_action(self, state):
        return np.array([agent.sample_behavior_action(state)
                         for agent in self.agents])
    
    def sample_target_action(self, state):
        return np.array([agent.sample_target_action(state)
                         for agent in self.agents])
    
    def get_rhos(self):
        return np.array([agent.get_rho() for agent in self.agents])
    
    def set_rhos(self, rhos):
        for i in range(self.num_agents):
            self.agents[i].set_rho(rhos[i])



       
# end
