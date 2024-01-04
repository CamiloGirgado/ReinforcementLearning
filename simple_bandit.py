import numpy

class MultiArmBandit:

    def __init__(self):
        self.bandit = [0.2, 0.0, 0.1, -4.0]
        self.num_actions = 4

    def pull(self,arm):
        return 1 if numpy.random.randn(1) > self.bandit[arm] else -1

# Main script for user interaction
def main():
    bandit = MultiArmBandit()

    while True:
        print("\nChoose an action (0 to {}), or '1' to quit".format(bandit.num_actions-1))
        user_input = input("> ")

        if user_input.lower() == 'q':
            break
                
        try:
            chosen_arm = int(user_input)
            if 0 <= chosen_arm < bandit.num_actions:
                reward = bandit.pull(chosen_arm)
                print(f"Chosen arm: {chosen_arm}, Reward: {reward}")
            else:
                print("Invalid input. Please choose a valid action")
            
        except ValueError:
            print("Invalid input. Please enter a valid integer or 'q' to Quit")

if __name__ == "__main__":
    main()