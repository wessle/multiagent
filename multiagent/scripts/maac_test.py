import numpy as np
import matplotlib
import time
import yaml
import os
from datetime import datetime
from matplotlib import pyplot as plt
from shutil import copyfile

import multiagent.core.envs as envs
import multiagent.core.agents as agents
import multiagent.core.runner as runner
import multiagent.core.consensus as consensus


if __name__ == '__main__':
    
    # Read config.
    config = 'maac_config.yml'
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
    min_reward_mean = config['min_reward_mean']
    max_reward_mean = config['max_reward_mean']
    var = config['reward_var']
    
    # Episode and other parameters.
    episode_length = config['episode_length']
    num_episodes = config['num_episodes']
        
    # Agent and training parameters.
    policy_units = config['policy_units']
    value_units = config['value_units']
    actor_lr = config['actor_lr']
    critic_lr = config['critic_lr']
    lambda_pi = config['lambda_pi']
    lambda_v = config['lambda_v']
    gamma = config['gamma']
    cov = config['cov']
    clip_grads = config['clip_grads']
    
    # Set up experiment directory.
    experiment_dir = \
        f"{config['results_path']}/{config['model']}-{datetime.now():%Y-%m-%d_%H:%M:%S}"
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    if config_path is not None:
        copyfile(config_path, f"{experiment_dir}/{config_path}")
    
    
    # Seed, prepare env, prepare consensus matrix.
    np.random.seed(env_seed)
    env = envs.ConvergenceTest(num_states, num_state_features, num_agents,
                               num_actions, min_reward_mean, max_reward_mean)
    env.init_features()
    env.init_transition_matrix()
    c1 = consensus.ConsensusCommunicator(num_agents)
    c1.random_matrix()
    
    print('Matrix created')
    
    # Set up runner.
    np.random.seed(agent_seed)
    runner = runner.MAACRunner(env, c1)
    runner.specify_episode(episode_length, num_episodes)
    runner.init_agents(policy_units, value_units, actor_lr, critic_lr,
                       lambda_pi, lambda_v, gamma, cov, clip_grads)
    
    
    # Do the test.
    start = time.time()
    rewards = runner.train()
    end = time.time()
    print('Completed {} episodes in {:.2f}m'.format(
            num_episodes, (end-start)/60))
        
    # Save rewards and a figure.
    rewards_path = f"{experiment_dir}/rewards.npy"
    np.save(rewards_path, rewards)
    plot_path = f"{experiment_dir}/plot.png"    
    matplotlib.use('Agg')
    plt.plot(np.arange(num_episodes), rewards)
    plt.savefig(plot_path)
    plt.close()
