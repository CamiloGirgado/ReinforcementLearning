# This class is made to create a running average for reward estimates. 
# The choose_action method selects the action with the highest estimated value, 
# and the update_q_estimate method updates the Q estimate using the provided formula.

import random
import matplotlib.pyplot as plt

class KArmBandit:
    def __init__(self, k, step_size, epsilon):
        self.k = k
        self.step_size = step_size
        self.epsilon = epsilon
        self.q_estimates = [0.0] * k
        self.action_counts = [0] * k

    def choose_action(self):
        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            return random.randint(0, self.k - 1)
        else:
            # Choose action with highest estimated value
            return self.q_estimates.index(max(self.q_estimates))

    def update_q_estimate(self, chosen_action, reward):
        # Update the Q estimate using the formula
        old_estimate = self.q_estimates[chosen_action]
        self.action_counts[chosen_action] += 1
        step_size = 1 / self.action_counts[chosen_action]
        self.q_estimates[chosen_action] = old_estimate + step_size * (reward - old_estimate)


# Example of using the KArmBandit class
def main():
    # Initialize a 3-arm bandit with a step size of 0.1
    bandit = KArmBandit(k=10, step_size=0.1, epsilon = 0.0)
    bandit_epsilon_001 = KArmBandit(k=10, step_size=0.1, epsilon=0.01)
    bandit_epsilon_01 = KArmBandit(k=10, step_size=0.1, epsilon=0.1)

    rewards = []  
    rewards_epsilon_001 = [] 
    rewards_epsilon_01 = []

    # Simulate 1000 steps
    for _ in range(1000):
        # Choose an action
        chosen_action = bandit.choose_action()
        reward = random.random()
        bandit.update_q_estimate(chosen_action, reward)
        rewards.append(sum(bandit.q_estimates) / len(bandit.q_estimates))

        chosen_action_epsilon_001 = bandit_epsilon_001.choose_action()
        reward_epsilon_001 = random.random()
        bandit_epsilon_001.update_q_estimate(chosen_action_epsilon_001, reward_epsilon_001)
        rewards_epsilon_001.append(sum(bandit_epsilon_001.q_estimates) / len(bandit_epsilon_001.q_estimates))

        chosen_action_epsilon_01 = bandit_epsilon_01.choose_action()
        reward_epsilon_01 = random.random()
        bandit_epsilon_01.update_q_estimate(chosen_action_epsilon_01, reward_epsilon_01)
        rewards_epsilon_01.append(sum(bandit_epsilon_01.q_estimates) / len(bandit_epsilon_01.q_estimates))

        # Update the Q estimate based on the chosen action and reward
        bandit.update_q_estimate(chosen_action, reward)

    # Print the final Q estimates
    print("Final Q Estimates:", bandit.q_estimates)
    print("Final Q Estimates (Epsilon=0.01):", bandit_epsilon_001.q_estimates)
    print("Final Q Estimates (Epsilon=0.1):", bandit_epsilon_01.q_estimates)

    # Plotting the results
    plt.plot(rewards, label='Regular Bandit')
    plt.plot(rewards_epsilon_001, label='Epsilon=0.01')
    plt.plot(rewards_epsilon_01, label='Epsilon=0.1')
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
