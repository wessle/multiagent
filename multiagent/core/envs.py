import numpy as np


class OneDimGridworld:
    def __init__(self, num_tiles, num_actions=3, reward_function=None,
                 goal_tile=None):
        self.num_tiles = num_tiles
        self.num_actions = num_actions    # 0 -> left, 1 -> stay, 2 -> right
        self.state = np.random.randint(self.num_tiles)
        self.observation_dim = 1
        
        # These normalizers are redundant here, but in more complicated envs
        # they will be needed.
        self.state_normalizer = self.num_tiles
        self.action_normalizer = self.num_actions
        
        if reward_function is not None:
            self.reward_function = reward_function
        else:
            self.reward_function = self._default_reward
        
        if goal_tile is not None:
            self.goal_tile = goal_tile
        else:
            self.goal_tile = np.random.randint(self.num_tiles)

    def _default_reward(self, state):
        return (-1)**(1 + (state == self.goal_tile))
        
    def step(self, action):
        real_action = action - 1
        self.state = (self.state + real_action) % self.num_tiles
        return self.reward_function(self.state), self.state
    
    def reset(self, init_state=None):
        if init_state is not None:
            self.state = init_state
        else:
            self.state = np.random.randint(self.num_tiles)
            
            
# Multi-agent environments.
class MAEnvironment:
    '''Empty parent environment.'''
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.observation_dim = None    # To be defined.
        self.curr_state = None
        self.curr_action = None
        
    def reset(self):
        # Reset the environment by choosing an initial state.
        pass
        
    def _gather_action(self, i, action):
        # Gather action from agent i.
        pass
        
    def _transition(self):
        # Once actions have been gathered from each agent, transition to
        # next state based on curr_state and curr_action.
        pass
    
    def _agent_reward(self, agent):
        # Return reward of agent given curr_state and curr_action.
        pass
    
    
class ConvergenceTest(MAEnvironment):
    '''An arbitrary environment for use in testing convergence.'''
    def __init__(self, num_states, num_state_features, num_agents,
                 num_actions, min_reward=0, max_reward=4):
        self.num_states = num_states
        self.observation_dim = num_state_features
        self.num_agents = num_agents
        self.num_actions = num_actions
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.feature_matrix = None
        self.transition_matrix = None
        self.curr_state = None
        self.curr_state_index = None
        self.curr_action = None
        
        # Uniformly generate rewards for each state-action
        # pair for each agent.
        self.rewards = np.zeros(
                (self.num_states, self.num_agents),
                dtype=(int, self.num_actions))
        for i in range(self.num_states):
            for j in range(self.num_actions):
                self.rewards[i][j] = tuple(np.random.uniform(
                        self.min_reward, self.max_reward, self.num_actions))
        
    def init_features(self, feature_matrix=None):
        if feature_matrix is not None:
            self.feature_matrix = feature_matrix
        else:
            rank = 0
            while rank < self.observation_dim:
                self.feature_matrix = np.random.random(
                        (self.num_states, self.observation_dim))
                rank = np.linalg.matrix_rank(self.feature_matrix)
                
    def init_transition_matrix(self, transition_matrix=None):
        if transition_matrix is not None:
            self.transition_matrix = transition_matrix
        else:
            self.transition_matrix = np.random.random(
                    (self.num_states, self.num_states))
            sums = self.transition_matrix.sum(axis=1)
            self.transition_matrix = \
                self.transition_matrix / sums[:, np.newaxis]
                
    def _get_state(self, index):
        return self.feature_matrix[index]
                
    def reset(self):
        self.curr_state_index = np.random.randint(0, self.num_states)
        self.curr_state = self._get_state(self.curr_state_index)
        self.curr_action = np.zeros(self.num_agents)
    
    def _gather_action(self, action):
        self.curr_action = action
    
    def _transition(self):
        self.curr_state_index = np.random.choice(
                self.num_states,
                p=self.transition_matrix[self.curr_state_index])
        self.curr_state = self._get_state(self.curr_state_index)
        
    def _agent_reward(self, agent):
        # Generate a given agent's reward.
        i = self.curr_state_index
        k = self.curr_action[agent]
        return self.rewards[i][agent][k]
    
    def _reward(self):
        # Return vector of all agents' rewards.
        rewards = np.array([self._agent_reward(agent)
            for agent in np.arange(self.num_agents)])
        return rewards
    
    def step(self, action):
        self._gather_action(action)
        self._transition()
        return self._reward(), self.curr_state
        
        
class MAGridworld(MAEnvironment):
    '''Gridworld environment where the goal is to have all agents occupy
    the same gridpoint.'''
    def __init__(self, num_agents, grid_dim, min_reward_distance=1):
        self.num_actions = 5    # 0, 1, 2, 3, 4 = stay, left, right, up, down
        self.num_agents = num_agents
        self.observation_dim = 2 * self.num_agents
        self.grid_dim = grid_dim
        self.min_reward_distance = min_reward_distance
        self.curr_state = None
        self.curr_action = None
        
    def _flat_state(self):
        return self.curr_state.flatten()
        
    def _gather_action(self, action):
        # Gather all actions at once in a single vector.
        self.curr_action = action
        
    def _convert_action(self, action):
        if action == 0:
            return np.zeros(2)
        elif action == 1:
            return np.array([-1, 0])
        elif action == 2:
            return np.array([1, 0])
        elif action == 3:
            return np.array([0, 1])
        else:
            return np.array([0, -1])
        
    def _project_state(self):
        self.curr_state = np.clip(self.curr_state, 0, self.grid_dim-1)
        
    def _transition(self):
        action = np.array([self._convert_action(action)
                          for action in self.curr_action])
        self.curr_state = (self.curr_state + action).astype(int)
        self._project_state()
        
    def _agent_reward(self, i):
        # Compute vector of distances from agent i to other agents,
        # then return reward.
        vecs = self.curr_state \
               - self.curr_state[i] * np.ones(self.curr_state.shape)
        vecs[i] = 10*np.ones(2)
        distances = np.linalg.norm(vecs, ord=np.inf, axis=1)
        rewards = np.array([1 for val in distances if val <= 1])
        return sum(rewards)
        
    def _reward(self):
        # Return reward for all agents based on current positions.
        rewards = np.array([self._agent_reward(agent)
            for agent in np.arange(self.num_agents)])
        return rewards
    
    def step(self, action):
        self._gather_action(action)
        self._transition()
        return self._reward(), self._flat_state()
    
    def reset(self):
        self.curr_state = np.random.randint(0, self.grid_dim,
                                            (self.num_agents, 2))
        self.curr_action = np.zeros(self.num_agents)
        
        
        
        
        
        


# end