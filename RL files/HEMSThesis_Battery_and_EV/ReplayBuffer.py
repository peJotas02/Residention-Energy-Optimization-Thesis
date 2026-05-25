import numpy as np
import random
from collections import deque
import torch
from torch.utils.data import TensorDataset

class ReplayBuffer():
    def __init__(self, buffer_capacity):
        self.buffer = deque(maxlen=buffer_capacity)
        # list of lists: [list[(s,a,r,ns,d)]]

    def push(self, transitions_list):
        self.buffer.append(list(transitions_list))


    def sample_MC(self, is_elite):
        return (self.elite_buffer.sample() if is_elite else random.choice(self.buffer))


    """def sample(self, batch_size, t):

        buffer_split = 0
        sample_size_elite = int(max(batch_size, len(self.buffer) // 10) * buffer_split) #sample size for each buffer (elite and standard)
        sample_size_standard = int(max(batch_size, len(self.buffer) // 10) * (1-buffer_split)) #sample size for each buffer (elite and standard)

        sample_standard = random.sample(self.buffer, sample_size_standard)
        sample_elite = [random.choice(self.elite_buffer.get_list()) for _ in range(sample_size_elite)]

        sample = sample_elite + sample_standard
        random.shuffle(sample)
        states, actions, rewards, next_states, dones = map(np.array, zip(*sample))

        states = torch.as_tensor(states, dtype=torch.float32)
        actions = torch.as_tensor(actions, dtype=torch.long)
        rewards = torch.as_tensor(rewards, dtype=torch.float32)
        next_states = torch.as_tensor(next_states, dtype=torch.float32)
        dones = torch.as_tensor(dones, dtype=torch.float32)

        buffer_batches =  torch.utils.data.DataLoader( TensorDataset(states, actions, rewards, next_states, dones), batch_size=batch_size, shuffle=True, drop_last=True )

        return buffer_batches"""

    def __len__(self):
        return len(self.buffer)
