# import re
# import matplotlib.pyplot as plt
#
# # Define a regular expression to extract the AVG Reward tot value from each line
# avg_reward_regex = r" AVG Reward for the episode ([\-0-9\.]+)"
#
# # Initialize lists to store the episode numbers and AVG loss values
# episode_nums = []
# avg_losses = []
#
# # Read the file and extract the AVG Reward tot value and episode number from each line
# ep_num = 0
# with open("evaluation_to_plot", "r") as f:
#     for line in f:
#         if not line.startswith("Reward list"):
#             avg_reward_match = re.search(avg_reward_regex, line)
#             if avg_reward_match:
#                 ep_num += 1
#                 episode_num = ep_num
#                 avg_loss = float(avg_reward_match.group(1))
#                 episode_nums.append(episode_num)
#                 avg_losses.append(avg_loss)
#
# # Create a scatter plot of the AVG loss over time
# plt.scatter(episode_nums, avg_losses, s=5)
# plt.xlabel("Episode")
# plt.ylabel("AVG loss tot")
# plt.show()


import re
import matplotlib.pyplot as plt

# Define a regular expression to extract the AVG Reward tot value from each line
avg_reward_regex = r"AVG Return ([\-0-9\.]+)"

# Initialize a list to store the AVG Reward tot values
avg_rewards = []

# Read the file and extract the AVG Reward tot value from each line
with open("evaluation_to_plot", "r") as f:
    for line in f:
        if not line.startswith("Reward list"):
            avg_reward_match = re.search(avg_reward_regex, line)
            if avg_reward_match:
                avg_rewards.append(float(avg_reward_match.group(1)))

# Create a line plot of the AVG Reward tot over time
plt.plot(range(1, len(avg_rewards) + 1), avg_rewards)
plt.xlabel("Episode")
plt.ylabel("AVG Reward tot")
plt.show()


