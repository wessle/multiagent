import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import multiagent.core.agents as agents

# TODO: make a generic Runner parent class for all the other runners.

class ACRunner:
    def __init__(self, env):
        self.env = env
        self.agent = None
        self.episode_length = 0
        self.num_episodes = 0
        self.label = None
        
    def specify_episode(self, episode_length, num_episodes):
        self.episode_length = episode_length
        self.num_episodes = num_episodes
        
    def init_agent(self, policy_units, value_units, actor_stepsize,
                   critic_stepsize, policy_trace, value_trace, gamma, cov,
                   clip_grads=False, normalize=False):
        self.env.reset()
        self.agent = agents.SimpleNeuralACAgent(
                policy_units, value_units, self.env.observation_dim,
                self.env.num_actions, actor_stepsize, critic_stepsize,
                policy_trace, value_trace, gamma, cov, clip_grads, normalize)
        
        if normalize:
            self.agent.register_normalizers(self.env.state_normalizer,
                                            self.env.action_normalizer)
        
        # TODO: remove label; this information should be part of external
        # config files and plotting utils.
        self.label = ('a/c units: {:d}/{:d}; '
                'stepsizes: {:.4f}/{:.4f}; '
                'traces: {:.1f}/{:.1f}; '
                'gamma: {:.2f}; cov: {:.2f}').format(
                policy_units, value_units, actor_stepsize,
                critic_stepsize, policy_trace, value_trace, gamma, cov)
        
    def run(self):
        self.env.reset()
        episode_rewards = []
        for i in range(self.num_episodes):
            total_reward = 0
            for _ in range(self.episode_length):
                action = self.agent.sample_action(self.env.state)
                reward, next_state = self.env.step(action)
                self.agent.update(next_state, reward)
                total_reward += reward
            episode_rewards.append(total_reward)
            total_reward = 0
            
        return episode_rewards
         
    # TODO: remove plotting from inside the Runner; it should be in
    # external utils.
    def _general_make_plot(self, episode_rewards, plot_name, plot_label):
        matplotlib.use('Agg')
        plt.plot(np.arange(self.num_episodes), episode_rewards)
        plt.xlabel(plot_label)
        plt.savefig(plot_name)
        plt.close()
        
    def make_plot(self, total_rewards, plot_name):
        self._general_make_plot(total_rewards, plot_name, self.label)


class LinearACRunner(ACRunner):
    def __init__(self, env):
        
        ACRunner.__init__(self, env)
        
    def init_agent(self, actor_stepsize, critic_stepsize,
                   policy_trace, value_trace, gamma, cov, clip_grads):
        self.env.reset()
        self.agent = agents.LinearACAgent(
                self.env.observation_dim, 1,
                self.env.num_actions,
                actor_stepsize, critic_stepsize, policy_trace, value_trace,
                gamma, cov, cov, clip_grads)
            
        self.label = ('LinearACAgent; '
                'stepsizes: {:.4f}/{:.4f}; '
                'traces: {:.1f}/{:.1f}; '
                'gamma: {:.2f}; cov: {:.2f}').format(
                actor_stepsize, critic_stepsize,
                policy_trace, value_trace, gamma, cov)
        
        
class OPACRunner(ACRunner):
    '''Make a runner for an off-policy actor-critic agent from ACRunner.'''
    def __init__(self, env):
        self.trial_length = 0
        ACRunner.__init__(self, env)
        
    def specify_episode(self, episode_length, num_episodes, trial_length):
        self.episode_length = episode_length
        self.num_episodes = num_episodes
        self.trial_length = trial_length
        
    def init_agent(self, num_target_policy_units,
                   num_target_value_func_units,
                   actor_stepsize, critic_stepsize, lambda_pi, lambda_v,
                   gamma, cov=1, clip_grads=False, normalize=False):
        self.env.reset()
        self.agent = agents.SimpleNeuralOPACAgent(
                num_target_policy_units, num_target_value_func_units,
                self.env.observation_dim,
                self.env.num_actions, actor_stepsize, critic_stepsize,
                lambda_pi, lambda_v, gamma, cov, clip_grads, normalize)
        
        if normalize:
            self.agent.register_normalizers(self.env.state_normalizer,
                                            self.env.action_normalizer)
        
        self.label = ('a/c units: {:d}/{:d}; '
                'stepsizes: {:.4f}/{:.4f}; '
                'traces: {:.1f}/{:.1f}; '
                'gamma: {:.2f}; cov: {:.2f}').format(
                num_target_policy_units, num_target_value_func_units,
                actor_stepsize, critic_stepsize, lambda_pi, lambda_v,
                gamma, cov)
        
    def train(self):
        self.env.reset()
        for i in range(self.num_episodes):
            for _ in range(self.episode_length):
                action = self.agent.sample_behavior_action(self.env.state)
                reward, next_state = self.env.step(action)
                self.agent.update(next_state, reward)
                
    def test_target_policy(self, trial_length):
        self.env.reset()
        episode_rewards = []
        for i in range(trial_length):
            total_reward = 0
            for _ in range(self.episode_length):
                action = self.agent.sample_target_action(self.env.state)
                reward, _ = self.env.step(action)
                total_reward += reward
            episode_rewards.append(total_reward)
            
        return episode_rewards
    
    def _general_make_plot(self, episode_rewards, plot_name, plot_label):
        matplotlib.use('Agg')
        plt.plot(np.arange(self.trial_length), episode_rewards)
        plt.xlabel(plot_label)
        plt.savefig(plot_name)
        plt.close()
        
    def make_plot(self, total_rewards, plot_name):
        self._general_make_plot(total_rewards, plot_name, self.label)
        
        
class DiagnosticOPACRunner(OPACRunner):
    def __init__(self, env):
        OPACRunner.__init__(self, env)
        
    def specify_episode(self, episode_length, num_episodes):
        self.episode_length = episode_length
        self.num_episodes = num_episodes
        
    def train(self):
        self.env.reset()
        for i in range(self.num_episodes):
            for _ in range(self.episode_length):
                action = self.agent.sample_behavior_action(self.env.state)
                reward, next_state = self.env.step(action)
                self.agent.update(next_state, reward)
            
            print('Episode {} correct tile: {}'.format(i, self.env.goal_tile))
            for tile in range(1, self.env.num_tiles+1):
                print('State {} probs:'.format(tile))
                print(self.agent.pi._get_probs(tile))
                print('\n')
                
           
class QRunner(ACRunner):
    '''Make a runner for the Q-learning agent by overriding the init_agent()
    method from ACRunner.'''
    def __init__(self, env):
        ACRunner.__init__(self, env)
   
    def init_agent(self, alpha, epsilon, gamma, output_dim, hidden_units,
                        input_dim, num_actions):
        self.env.reset()
        self.agent = agents.QLearningAgent(
                alpha, epsilon, gamma, output_dim, hidden_units,
                input_dim, num_actions)
        self.label = ('units: {:d}; '
                'stepsize: {:.4f}; epsilon: {:.2f} '
                'gamma: {:.2f}').format(
                hidden_units, alpha, epsilon, gamma)
        
        
# Multi-agent Runners
                
class MAACRunner:
    '''Take an MAEnvironment and ConsensusCommunicator for critic consensus,
    then fill an array with a bunch of ACAgents. Use run method to train.'''
    def __init__(self, env, critic_consensus):
        self.env = env
        self.metaagent = None
        self.critic_consensus = critic_consensus
        self.episode_length = 0
        self.num_episodes = 0
        
    def specify_episode(self, episode_length, num_episodes):
        self.episode_length = episode_length
        self.num_episodes = num_episodes
        
    def init_agents(self, policy_units, value_units, actor_stepsize,
                    critic_stepsize, policy_trace, value_trace, gamma, cov=1.0,
                    clip_grads=False):
        self.env.reset()
        agent = agents.SimpleNeuralACAgent(
                policy_units, value_units, self.env.observation_dim,
                self.env.num_actions, actor_stepsize, critic_stepsize,
                policy_trace, value_trace, gamma, cov, clip_grads)
        self.metaagent = agents.MetaACAgent(agent, self.env.num_agents)
        
    def train(self):
        self.env.reset()
        episode_rewards = []
        for i in range(self.num_episodes):
            total_reward = 0
            for _ in range(self.episode_length):
                # Perform consensus on value parameters.
                self.metaagent.set_value_params(
                        self.critic_consensus.consensus_step(
                                self.metaagent.get_value_params()))
                
                # Update.
                action = self.metaagent.sample_action(self.env.curr_state)
                rewards, next_state = self.env.step(action)
                self.metaagent.update(next_state, rewards)
                total_reward += np.average(rewards)
                
            episode_rewards.append(total_reward)
            total_reward = 0
            
        return episode_rewards
    
    def get_state_value_estimates(self, states):
        return np.array([self.metaagent.get_average_state_value(state)
                         for state in states])
    

class LinearMAACRunner(MAACRunner):
    '''MAACRunner using linear function approximation.'''
    def __init__(self, env, critic_consensus):
        MAACRunner.__init__(self, env, critic_consensus)
        
    def init_agents(self, actor_stepsize, critic_stepsize,
                    policy_trace, value_trace, gamma, cov=1.0,
                    clip_grads=False):
        self.env.reset()
        agent = agents.LinearACAgent(
                self.env.observation_dim, 1, self.env.num_actions,
                actor_stepsize, critic_stepsize,
                policy_trace, value_trace,
                gamma, cov, cov, clip_grads)
        self.metaagent = agents.MetaACAgent(agent, self.env.num_agents)
        
        
class MAOPACRunner(MAACRunner):
    '''Same as MAACRunner, except also take a ConsensusCommunicator for the
    inner consensus loop, and use OPACAgents.'''
    def __init__(self, env, critic_consensus, rho_consensus):
        
        MAACRunner.__init__(self, env, critic_consensus)
        self.rho_consensus = rho_consensus
        self.step_counter = 0
        
        self.custom_stepsize_functions = False
        self.actor_stepsize_function = None
        self.critic_stepsize_function = None
        
    def init_agents(self, policy_units, value_units, actor_stepsize,
                    critic_stepsize, lambda_pi, lambda_v, gamma, cov=1.0,
                    clip_grads=False, clip_rho=True):
        self.env.reset()
        agent = agents.SimpleNeuralOPACAgent(
                policy_units, value_units,
                self.env.observation_dim,
                self.env.num_actions, actor_stepsize, critic_stepsize,
                lambda_pi, lambda_v, gamma, cov, clip_grads, clip_rho)
        self.metaagent = agents.MetaOPACAgent(agent, self.env.num_agents)
        
    def init_stepsize_functions(self, actor_stepsize_function,
                                critic_stepsize_function):
        self.custom_stepsize_functions = True
        self.actor_stepsize_function = actor_stepsize_function
        self.critic_stepsize_function = critic_stepsize_function
        
    def train(self):
        self.env.reset()
        for i in range(self.num_episodes):
            for _ in range(self.episode_length):                
                # Perform consensus on value params.
                self.metaagent.set_value_params(
                        self.critic_consensus.consensus_step(
                                self.metaagent.get_value_params()))
                
                # Choose action.
                action = self.metaagent.sample_behavior_action(
                        self.env.curr_state)
                rewards, next_state = self.env.step(action)
                
                # Perform consensus on rho.
                self.metaagent.set_rhos(
                        np.exp(self.rho_consensus.multistep_consensus(
                                np.log(self.metaagent.get_rhos()))))
                
                # Perform updates.
                self.step_counter += 1
                if self.custom_stepsize_functions:
                    self.metaagent.change_stepsizes(
                            self.actor_stepsize_function(self.step_counter),
                            self.critic_stepsize_function(self.step_counter))
                self.metaagent.update(next_state, rewards)
        
    def test_target_policy(self, trial_length):
        self.env.reset()
        episode_rewards = []
        for i in range(trial_length):
            total_reward = 0
            for _ in range(self.episode_length):
                action = self.metaagent.sample_target_action(
                        self.env.curr_state)
                rewards, _ = self.env.step(action)
                total_reward += np.average(rewards)
            episode_rewards.append(total_reward)
            
        return episode_rewards


class LinearMAOPACRunner(MAOPACRunner):
    '''MAOPACRunner with linear function approximation.'''
    def __init__(self, env, critic_consensus, rho_consensus):
        
        MAOPACRunner.__init__(self, env, critic_consensus, rho_consensus)
        
    def init_agents(self, policy_units,
                    state_vector_len, num_actions, actor_stepsize,
                    critic_stepsize, lambda_pi, lambda_v, gamma, cov=1.0,
                    clip_grads=False, clip_rho=True):
        self.env.reset()
        agent = agents.SimpleNNPolicyLinearOPACAgent(
                policy_units, state_vector_len, num_actions, actor_stepsize,
                critic_stepsize, lambda_pi, lambda_v, gamma, cov,
                clip_grads, clip_rho)
        self.metaagent = agents.MetaOPACAgent(agent, self.env.num_agents)



# end
