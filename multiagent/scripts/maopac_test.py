# Linear version of the MAOPAC test.
import numpy as np
import matplotlib
import time
import yaml
import os
from datetime import datetime
from matplotlib import pyplot as plt
from shutil import copyfile
from shutil import rmtree

import multiagent.core.envs as envs
import multiagent.core.agents as agents
import multiagent.core.runner as runner
import multiagent.core.consensus as consensus


np.seterr(divide='ignore', invalid='ignore')


UPDATE_INTERVAL = 10


if __name__ == '__main__':
    
    # Read config.
    config = 'maopac_config.yml'
    config_path = None
    if isinstance(config, str):
        config_path = config
        with open(config_path) as f:
            config = yaml.load(f)
    else:
        raise ValueError('config should be a string')
    
    # Environment parameters and seed.
    env_seed = config['env_seed']
    agent_seed = config['agent_seed']
    
    num_states = config['num_states']
    num_state_features = config['num_state_features']
    num_agents = config['num_agents']
    num_actions = config['num_actions']
    min_reward = config['min_reward']
    max_reward = config['max_reward']
    
    # Episode and other parameters.
    episode_length = config['episode_length']
    num_episodes = config['num_episodes']
    num_trains = config['num_trains']
        
    # Agent and training parameters.
    policy_units = config['policy_units']
    actor_lr = config['actor_lr']
    critic_lr = config['critic_lr']
    lambda_pi = config['lambda_pi']
    lambda_v = config['lambda_v']
    gamma = config['gamma']
    cov = config['cov']
    clip_grads = config['clip_grads']
    clip_rho = config['clip_rho']
    
    # Decreasing stepsizes.
    decreasing_stepsizes = config['decreasing_stepsizes']
    actor_stepsize_exponent = config['actor_stepsize_exponent']
    critic_stepsize_exponent = config['critic_stepsize_exponent']
    
    # Edge probability for communication graph.
    edge_probability = config['edge_probability']
    
    # Save results or not?
    save_results = config['save_results']
    
    # rho consensus
    multistep_consensus_steps = config['multistep_consensus_steps']
    rho_consensus = config['rho_consensus']
    
    # Set up experiment directory.
    experiment_dir = \
        f"{config['results_path']}/{config['model']}-{datetime.now():%Y-%m-%d_%H:%M:%S}"
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    if config_path is not None:
        copyfile(config_path, f"{experiment_dir}/{config_path}")
    
    
    # Seed, prepare env, prepare consensus matrices.
    np.random.seed(env_seed)
    env = envs.ConvergenceTest(num_states, num_state_features, num_agents,
                               num_actions, min_reward, max_reward)
    env.init_features()
    env.init_transition_matrix()
    c1 = consensus.ConsensusCommunicator(num_agents, edge_probability)
    c1.random_stochastified_laplacian()
    
    c2 = consensus.ConsensusCommunicator(num_agents,
                                         multistep_consensus_steps)
    if rho_consensus == 'perfect':
        c2.set_matrix(1/num_agents * np.ones((num_agents, num_agents)))
    else:
        c2.set_matrix(c1.consensus_matrix)
    
    print('Consensus matrices created')
    
    # Set up the runner.
    np.random.seed(agent_seed)
    runner = runner.LinearMAOPACRunner(env, c1, c2)
    runner.specify_episode(episode_length, num_episodes)
    runner.init_agents(policy_units, num_state_features, num_actions,
                       actor_lr, critic_lr, lambda_pi, lambda_v, gamma, cov,
                       clip_grads, clip_rho)
    
    # Set decreasing stepsize functions.
    if decreasing_stepsizes:
        runner.init_stepsize_functions(
                lambda t: (1/t)**actor_stepsize_exponent,
                lambda t: (1/t)**critic_stepsize_exponent)
    
    
    #############
    #### RUN ####
    #############
    
    
    rewards = []
    start = time.time()
    
    for i in range(num_trains):
        
        if i % UPDATE_INTERVAL == 0:
            print('Training run: {}'.format(i))
        runner.train()
        
        state_val_estimates = runner.get_state_value_estimates(
                runner.env.feature_matrix)
        rewards.append(np.average(state_val_estimates))
        
    end = time.time()
    print('Time elapsed: {:.2f}m'.format((end - start)/60))
    
    if save_results:
        # Save rewards and a figure.
        rewards_path = f"{experiment_dir}/rewards.npy"
        np.save(rewards_path, rewards)
        plot_path = f"{experiment_dir}/plot.png"    
        matplotlib.use('Agg')
        plt.plot(np.arange(num_trains), rewards)
        plt.savefig(plot_path)
        plt.close()
    elif not save_results:
        rmtree(experiment_dir)
