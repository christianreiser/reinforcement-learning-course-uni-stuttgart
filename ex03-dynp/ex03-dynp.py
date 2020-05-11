import gym
import numpy as np

# Init environment
env = gym.make("FrozenLake-v0")
# you can set it to deterministic with:
# env = gym.make("FrozenLake-v0", is_slippery=False)

# If you want to try larger maps you can do this using:
# random_map = gym.envs.toy_text.frozen_lake.generate_random_map(size=5, p=0.8)
# env = gym.make("FrozenLake-v0", desc=random_map)


# Init some useful variables:
n_states = env.observation_space.n
n_actions = env.action_space.n


def value_iteration():
    policy_best = np.zeros(n_states).astype(int)
    v_of_best_action = np.zeros(n_states)
    v_old = np.zeros(n_states)
    convergence_counter = 0
    theta = 1e-8
    gamma = 0.8

    while True:  # repeat as long as at least one value of a state can be improved within one iteration
        delta = 0  # checker for improvements across states
        convergence_counter += 1
        for state in range(n_states):
            v_of_a = np.zeros(n_actions)
            v_old[state] = v_of_best_action[state]
            for action in range(n_actions):  # iterate through all actions to find optimal action
                # env.P[s][a] is a list of transition tuples (prob, next_state, reward, done)
                # p is the probability of transitioning from state to next_state and receiving reward r given the action
                for p, next_state, r, done in env.P[state][action]:  # is_terminal
                    # sum over all next_states the (probability that it happens * reward)
                    v_of_a[action] += p * (r + gamma * v_old[next_state])
                if v_of_a[action] > v_of_best_action[state]:
                    v_of_best_action[state] = v_of_a[action]
                    policy_best[state] = action
            delta = max(delta, np.abs(v_of_best_action[state] - v_old[state]))  # max value improvement over all states

        if delta < theta:  # break if the change in value is less than the threshold (theta)
            break
    print('convergence counter:', convergence_counter)
    print('optimal value function:')
    print('',
          round(v_of_best_action[0], 3),
          round(v_of_best_action[1], 3),
          round(v_of_best_action[2], 3),
          round(v_of_best_action[3], 3), '\n',
          round(v_of_best_action[4], 3),
          round(v_of_best_action[5], 3),
          round(v_of_best_action[6], 3),
          round(v_of_best_action[7], 3), '\n',
          round(v_of_best_action[8], 3),
          round(v_of_best_action[9], 3),
          round(v_of_best_action[10], 3),
          round(v_of_best_action[11], 3), '\n',
          round(v_of_best_action[12], 3),
          round(v_of_best_action[13], 3),
          round(v_of_best_action[14], 3),
          round(v_of_best_action[15], 3), '\n')
    return policy_best


def main():
    # print the environment
    print("current environment: ")
    env.render()
    print("")

    # run the value iteration
    policy = value_iteration()
    print("Computed policy:")
    # print(policy)
    print('', policy[0], policy[1], policy[2], policy[3], '\n', policy[4], policy[5], policy[6], policy[7], '\n',
          policy[8], policy[9], policy[10], policy[11], '\n', policy[12], policy[13], policy[14], policy[15], '\n')

    # This code can be used to "rollout" a policy in the environment:
    """print ("rollout policy:")
    maxiter = 100
    state = env.reset()
    for i in range(maxiter):
        new_state, reward, done, info = env.step(policy[state])
        env.render()
        state=new_state
        if done:
            print ("Finished episode")
            break"""


if __name__ == "__main__":
    main()
