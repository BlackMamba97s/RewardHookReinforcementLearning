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


# update step using experience, to be seen again, need some tests
#     x_batch, action_batch, old_action_log_probs_batch, value_batch, returns_batch, advantages_batch = zip(*experiences)
#     x_batch = torch.stack(x_batch)
#     action_batch = torch.stack(action_batch)
#     old_action_log_probs_batch = torch.stack(old_action_log_probs_batch)
#     value_batch = torch.stack(value_batch)
#     returns_batch = torch.stack(returns_batch)
#     advantages_batch = torch.stack(advantages_batch)
#     _, action_log_probs_batch = policy_network.sample(x_batch)
#     values_batch = value_network(x_batch)
#
#     policy_loss, value_loss = compute_ppo_loss(action_log_probs_batch, values_batch, advantages_batch, returns_batch,
#                                                old_action_log_probs_batch, eps=0.2)
#     loss = policy_loss + value_loss
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
