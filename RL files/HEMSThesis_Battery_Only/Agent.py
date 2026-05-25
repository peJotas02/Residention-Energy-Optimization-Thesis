import numpy as np
import torch
import torch.optim as optim
from DQN import DQN
from ReplayBuffer import ReplayBuffer
import random
from torch import nn

class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        norm_factors,
        total_eps,
        lr=5e-5,
        gamma=1,
        epsilon_start=1.0,
        epsilon_end=0.05,
        buffer_size=(800 * 4), #best 800 * 4
        batch_size=1024,
        target_update_freq=(10),
        MC_freq_train=1,
        device="cpu"

    ):
        self.device = torch.device(device)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.norm_factors = torch.tensor(norm_factors, dtype=torch.float32, device=self.device)
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer_capacity = buffer_size
        self.buffer = ReplayBuffer(self.buffer_capacity)
        self.day_memory = []
        self.total_cost = 0

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.MC_freq_train = MC_freq_train
        self.num_train = 0
        self.elite_unchanged_for = 0
        self.prune = True
        self.total_eps = total_eps

        self.total_steps = 0

    def update_epsilon(self, epsilon):
        self.epsilon = (epsilon if epsilon > self.epsilon_end else self.epsilon_end)

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)
        #print('Model saved...')

    def load_model(self, path):
        print('Loading model...')
        state_dict = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(state_dict)

    def select_action(self, state, is_training=True):
        self.total_steps += 1

        if random.random() <= self.epsilon and is_training:
            return random.randrange(self.action_dim)
        else:
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            state_normalized = state / self.norm_factors
            state_normalized = state_normalized.unsqueeze(0)

            with torch.no_grad():
                q_values = self.policy_net(state_normalized)
            return torch.argmax(q_values).item()

    def store_step(self, state, action, reward, next_state, done):

        self.day_memory.append((state, action, reward, next_state, done))
        self.total_cost += reward

        if self.total_steps % 24 == 0:
            self.buffer.push(self.day_memory)
            self.day_memory = []
            self.total_cost = 0

    def train(self, num_epoch_MC, num_MC_samples):


        if (self.num_train % self.MC_freq_train == 0):
            for sample in range(num_MC_samples):
                MC_sample = self.buffer.sample_MC(is_elite=False)
                self.train_on_MC(MC_sample, num_epoch_MC)
        #else:
        #    self.train_on_TD(batch_size, num_epoch_TD)

        self.num_train += 1
        self.save_model('saved_model.pt')




    def train_on_MC(self, episode_transitions, num_epochs):
        """
        Monte Carlo update on a single episode.
        episode_transitions: list of (state, action, reward, next_state, done)
        """

        for _ in range(num_epochs):
            states, actions, rewards, next_states, dones = zip(*episode_transitions)

            # to tensors
            states = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
            actions = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
            rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)

            # compute full returns G_t
            G = 0.0
            returns = []
            for r in reversed(rewards.tolist()):
                G = r + self.gamma * G
                returns.insert(0, G)
            returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device).unsqueeze(1)

            # normalize states exactly like in train()
            states_normalized = states / self.norm_factors

            # Q(s,a) for taken actions
            q_taken = self.policy_net(states_normalized).gather(1, actions)

            # MSE between Q(s,a) and Monte Carlo returns
            loss = nn.MSELoss()(q_taken, returns)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    """
    def train_on_TD(self, batch_size, num_epoch):

        for _ in range(num_epoch):

            # select mini batch
            buffer_batches = self.buffer.sample(batch_size, self.total_steps)

            for states, actions, rewards, next_states, dones in buffer_batches:
                states = states.to(self.device)
                actions = actions.to(self.device).unsqueeze(1)
                rewards = rewards.to(self.device).unsqueeze(1)
                next_states = next_states.to(self.device)
                dones = dones.to(self.device).unsqueeze(1)

                # Normalize states
                states_normalized = states / self.norm_factors
                next_states_normalized = next_states / self.norm_factors

                # run mini batch on policy network
                curr_q_values = self.policy_net(states_normalized).gather(1, actions)

                # get values from the target network
                with torch.no_grad():
                    next_q_values = self.target_net(next_states_normalized).max(1)[0].unsqueeze(1)
                    target_q_values = rewards + self.gamma * next_q_values * (1 - dones)  # calculate target q values

                # get loss
                loss = nn.SmoothL1Loss()(curr_q_values, target_q_values)

                # optimize policy
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # update target
        if self.total_steps % (self.target_update_freq) == 0:  # update target after each training session
            self.target_net.load_state_dict(self.policy_net.state_dict())
        """


