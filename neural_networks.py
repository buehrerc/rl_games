"""
This File holds different kind of neural networks for different players
- TTTFlatNetwork
- TTTConvNetwork
"""
import torch
import torch.nn as nn


class TTTFlatNetwork(nn.Module):
    def __init__(self):
        super(TTTFlatNetwork, self).__init__()
        self.flat_seq = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 9),
            nn.Sigmoid()
        )
        self.architecture = 'flat'

    def forward(self, x):
        return self.flat_seq(x)


class TTTConvNetwork(nn.Module):
    def __init__(self):
        super(TTTConvNetwork, self).__init__()
        self.seq_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=2),
            nn.LeakyReLU(),
            nn.Flatten(),
        )
        self.seq_lin = nn.Sequential(
            nn.Linear(147, 147),
            nn.LeakyReLU(),
            nn.Linear(147, 75),
            nn.LeakyReLU(),
            nn.Linear(75, 9)
        )
        self.architecture = 'conv'

    def forward(self, x):
        x = self.seq_conv(x)
        return self.seq_lin(x.flatten())