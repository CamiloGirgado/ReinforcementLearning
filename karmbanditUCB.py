import math
import random
import matplotlib.pyplot as plt

class KArmBanditUCB:
    def __init__(self, k, c):
        self.k = k
        self.c = c  # Exploration parameter
        self.q_estimates = [0.0] * k
        self.action_counts = [0] * k
        self.total_steps = 0

    def choose_action(self):
        # UCB calculation to balance exploration and exploitation
        ucb_values = [estimate + self.c * math.sqrt(math.log(self.total_steps + 1) / max(count, 1)) for estimate, count in zip(self.q_estimates, self.action_counts)]
        
        # Choose action with highest UCB value
        return ucb_values.index(max(ucb_values))

    def update_q_estimate(self, chosen_action, reward):
        # Update the Q estimate using the formula
        old_estimate = self.q_estimates[chosen_action]
        self.action_counts[chosen_action] += 1
        self.total_steps += 1
        step_size = 1 / self.action_counts[chosen_action]
        self.q_estimates[chosen_action] = old_estimate + step_size * (reward - old_estimate)


# Example of using the KArmBanditUCB class and plotting results
def main():
    # Initialize a 3-arm bandit with exploration parameter c=2
    bandit_ucb = KArmBanditUCB(k=4, c=2)

    # Lists to store results for plotting
    steps = []
    q_estimates = [[] for _ in range(bandit_ucb.k)]

    # Simulate 1000 steps
    for step in range(1000):
        # Choose an action
        chosen_action = bandit_ucb.choose_action()

        # Simulate a reward (you would replace this with your actual reward function)
        # For simplicity, let's assume the reward is a random value between 0 and 1
        reward = random.random()

        # Update the Q estimate based on the chosen action and reward
        bandit_ucb.update_q_estimate(chosen_action, reward)

        # Store results for plotting
        steps.append(step)
        for arm in range(bandit_ucb.k):
            q_estimates[arm].append(bandit_ucb.q_estimates[arm])

    # Plot the results
    for arm in range(bandit_ucb.k):
        plt.plot(steps, q_estimates[arm], label=f'Arm {arm + 1}')

    plt.xlabel('Steps')
    plt.ylabel('Q Estimates')
    plt.title('UCB Bandit Problem')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
