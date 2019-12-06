# Tools for performing tests of (OP)AC.

import os
import time


DURATION = 1
FREQ = 440


def doLinACtest(counter, runrun,
            actor_stepsize, critic_stepsize,
            actor_trace, critic_trace,
            gamma, cov, clip_grads, play_sound):
    
    t0 = time.time()
    
    runrun.init_agent(actor_stepsize, critic_stepsize, actor_trace,
                      critic_trace, gamma, cov, clip_grads)
    
    results = runrun.run()
    
    plot_name = 'results/fig{:d}.png'.format(counter)
    runrun.make_plot(results, plot_name)
    
    t1 = time.time()
    
    print('Completed run {} in {:.1f}s.'.format(
            counter, t1-t0))
    
    if play_sound:
        os.system('play -nq -t alsa synth {} sine {}'.format(
                DURATION, FREQ))


def doACtest(counter, runrun, actor_hidden_units, critic_hidden_units,
            actor_stepsize, critic_stepsize,
            actor_trace, critic_trace,
            gamma, cov, clip_grads, normalize, play_sound):
    
    t0 = time.time()
    
    runrun.init_agent(actor_hidden_units, critic_hidden_units,
                      actor_stepsize, critic_stepsize,
                      actor_trace, critic_trace,
                      gamma, cov, clip_grads, normalize)
    results = runrun.run()
    
    plot_name = 'results/fig{:d}.png'.format(counter)
    runrun.make_plot(results, plot_name)
    
    t1 = time.time()
    
    print('Completed run {} in {:.1f}s.'.format(
            counter, t1-t0))
    
    if play_sound:
        os.system('play -nq -t alsa synth {} sine {}'.format(
                DURATION, FREQ))
        

def doOPACtest(counter, trial_length, runrun, actor_hidden_units,
               critic_hidden_units,
               actor_stepsize, critic_stepsize,
               actor_trace, critic_trace,
               gamma, cov, clip_grads, normalize, play_sound):
    
    t0 = time.time()
    
    runrun.init_agent(actor_hidden_units, critic_hidden_units,
                      actor_stepsize, critic_stepsize,
                      actor_trace, critic_trace,
                      gamma, cov, clip_grads, normalize)
    
    runrun.run()
    
    results = runrun.test_target_policy(trial_length)
    
    plot_name = 'results/fig{:d}.png'.format(counter)
    runrun.make_plot(results, plot_name)
    
    t1 = time.time()
    
    print('Completed run {} in {:.1f}s.'.format(
            counter, t1-t0))
    
    if play_sound:
        os.system('play -nq -t alsa synth {} sine {}'.format(
                DURATION, FREQ))
        

# TODO: make a general experiment runner function.