# coding: utf-8

import numpy as np

n_states = 500 # for Taxi-v3
n_actions = 6 # for Taxi-v3

def select_elites(states_batch, actions_batch, rewards_batch, percentile=50):
    reward_threshold = np. percentile(rewards_batch, percentile)
    elite_mask = np.asarray(rewards_batch) >= reward_threshold
    elite_states = np.asarray(states_batch, dtype='object')[elite_mask]
    elite_actions = np.asarray(actions_batch, dtype='object')[elite_mask]
    return np.concatenate(elite_states), np.concatenate(elite_actions)

def update_policy(elite_states, elite_actions, n_states=n_states, n_actions=n_actions):
    new_policy = np.zeros((n_states, n_actions))
    for state, action in zip(elite_states, elite_actions):
        new_policy[state, action] += 1
    absent_mask = new_policy.sum(axis=1) == 0
    new_policy[absent_mask] += 1
    new_policy /= new_policy.sum(axis=1, keepdims=True)
    return new_policy

def generate_session(env, policy, t_max=int(10**4)):
    states, actions = [], []
    total_reward = 0.
    s, info = env.reset()
    for t in range(t_max):
        a = np.random.choice(np.arange(len(policy[0])), p=policy[s])
        new_s, r, done, truncated, info = env.step(a)
        assert new_s is not None and r is not None and done is not None
        assert a is not None
        states.append(s)
        actions.append(a)
        total_reward += r
        s = new_s
        if done:
            break
    return states, actions, total_reward