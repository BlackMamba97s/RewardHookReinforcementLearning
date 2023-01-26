class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)),
                               size=batch_size,
                               replace=False)
        return [self.buffer[i] for i in idx]


    # to implement after the code run normally
# buffer = ReplayBuffer(capacity=10000)
#
# for episode in range(num_episodes):
#     for step in range(num_steps):
#         buffer.add((x, actions, action_log_probs, values, returns, advantages))
#         if len(buffer) >= batch_size:
#             experiences = buffer.sample(batch_size)
