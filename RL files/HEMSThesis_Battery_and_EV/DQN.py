import torch.nn as nn

class DQN(nn.Module):
    def __init__ (self, state_size, action_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_size),
        )

    def forward(self, x):
        return self.model(x)

