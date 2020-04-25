import numpy as np
import matplotlib.pyplot as plt
import random


class GaussianBandit:
    def __init__(self):
        self._arm_means = np.random.uniform(0., 1., 10)  # Sample some means
        self.n_arms = len(self._arm_means)
        self.rewards = []
        self.total_played = 0

    def reset(self):
        self.rewards = []
        self.total_played = 0

    def play_arm(self, a):
        reward = np.random.normal(self._arm_means[a], 1.)  # Use sampled mean and covariance of 1.
        self.total_played += 1
        self.rewards.append(reward)
        return reward


def greedy(bandit, timesteps):
    rewards = np.zeros(bandit.n_arms)
    n_plays = np.zeros(bandit.n_arms)
    Q = np.zeros(bandit.n_arms)
    possible_arms = range(bandit.n_arms)

    # init variables (rewards, n_plays, Q) by playing each arm once
    # Todo: why?
    for arm in possible_arms:
        bandit.play_arm(arm)

    # Main loop
    while bandit.total_played < timesteps:
        a = np.argmax(Q)  # choose greedy action
        rewards[a] += bandit.play_arm(a)  # update rewards
        n_plays[a] += 1
        Q = np.divide(rewards, n_plays)  # update value function


# epsilon greedy action selection (you can copy your code for greedy as a starting point)
def epsilon_greedy(bandit, timesteps):
    rewards = np.zeros(bandit.n_arms)
    n_plays = np.zeros(bandit.n_arms)
    Q = np.zeros(bandit.n_arms)
    possible_arms = range(bandit.n_arms)

    epsilon = 0.1



    while bandit.total_played < timesteps:
        # decide if next action is greedy
        random_float = np.random.rand()  # returns random float between 0 and 1
        if random_float >= epsilon:
            play_greedy = True
        else:
            play_greedy = False

        if play_greedy:
            a = np.argmax(Q)  # choose greedy action
        else:
            a = random.choice(possible_arms)  # random exploration action
        rewards[a] += bandit.play_arm(a)  # update rewards
        n_plays[a] += 1
        Q = np.divide(rewards, n_plays)  # update value function

def main():
    n_episodes = 10000  # TODO: set from 500 to 10000 to decrease noise in plot
    n_timesteps = 1000
    rewards_greedy = np.zeros(n_timesteps)
    rewards_egreedy = np.zeros(n_timesteps)

    for i in range(n_episodes):
        if i % 100 == 0:
            print ("current episode: " + str(i))

        b = GaussianBandit()  # initializes a random bandit
        greedy(b, n_timesteps)
        rewards_greedy += b.rewards

        b.reset()  # reset the bandit before running epsilon_greedy
        epsilon_greedy(b, n_timesteps)
        rewards_egreedy += b.rewards

    rewards_greedy /= n_episodes
    rewards_egreedy /= n_episodes
    plt.plot(rewards_greedy, label="greedy")
    print("Total reward of greedy strategy averaged over " + str(n_episodes) + " episodes: " + str(np.sum(rewards_greedy)))
    plt.plot(rewards_egreedy, label="e-greedy")
    print("Total reward of epsilon greedy strategy averaged over " + str(n_episodes) + " episodes: " + str(np.sum(rewards_egreedy)))
    plt.legend()
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.savefig('bandit_strategies.eps')
    plt.show()


if __name__ == "__main__":
    main()
