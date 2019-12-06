# Consensus objects.

import numpy as np
import networkx as nx


NORM_STEPS = 100

        
class ConsensusCommunicator:
    def __init__(self, num_agents, edge_probability=0.5, multistep_default=1):
        self.num_agents = num_agents
        self.agent_info = None    # This is a vector of vectors to be averaged.
        self.consensus_matrix = None    # Should be doubly stochastic.
        self.edge_probability = edge_probability
        self.multistep_default = multistep_default
        
    def set_matrix(self, matrix):
        self.consensus_matrix = matrix
        
    def _doubly_stochastify(self, matrix):
        A = matrix
        rsum = None
        csum = None
        counter = 0
        while (counter < NORM_STEPS) & ((np.any(rsum != 1)) | (np.any(csum != 1))):
            counter += 1
            A /= A.sum(0)
            A = A / A.sum(1)[:, np.newaxis]
            rsum = A.sum(1)
            csum = A.sum(0)
        return counter, A
    
    def _random_laplacian(self):
        G = nx.fast_gnp_random_graph(self.num_agents, self.edge_probability)
        L = nx.laplacian_matrix(G)
        return abs(L.toarray().astype(float))
    
    def _random_stochastified_laplacian(self):
        counter = NORM_STEPS
        while counter == NORM_STEPS:
            counter, A = self._doubly_stochastify(self._random_laplacian())
        return A
    
    def random_stochastified_laplacian(self):
        self.consensus_matrix = self._random_stochastified_laplacian()
        
    def random_matrix(self):
        # Generate random consensus matrix of size self.num_agents.
        # TODO: make this safe -- it currently may loop infinitely.
        A = np.random.random((self.num_agents, self.num_agents))
        _, self.consensus_matrix = self._doubly_stochastify(A)
        
    def _set_info(self, agent_info):
        self.agent_info = agent_info
        
    def _do_consensus(self):
        self.agent_info = np.dot(self.consensus_matrix, self.agent_info)
        
    def _get_info(self):
        return self.agent_info
    
    def consensus_step(self, agent_info):
        self._set_info(agent_info)
        self._do_consensus()
        return self._get_info()
    
    def multistep_consensus(self, agent_info,
                            num_steps=None):
        
        if num_steps is not None:
            pass
        else:
            num_steps = self.multistep_default
        
        self._set_info(agent_info)
        for _ in range(num_steps):
            self._do_consensus()
        return self._get_info()


class StochasticConsensusCommunicator(ConsensusCommunicator):
    def __init__(self,):
        pass






# end