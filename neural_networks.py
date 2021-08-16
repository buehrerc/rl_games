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
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 9),
        )
        self.architecture = 'flat'

    def forward(self, x):
        return self.flat_seq(x)


class TTTConvNetwork(nn.Module):
    def __init__(self):
        super(TTTConvNetwork, self).__init__()
        self.seq_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1))
        )
        self.seq_ = nn.Sequential(
            nn.Linear(864, 864),
            nn.LeakyReLU(),
            nn.Linear(864, 864),
            nn.LeakyReLU(),
            nn.Linear(864, 432),
            nn.LeakyReLU(),
            nn.Linear(432, 75),
            nn.LeakyReLU(),
            nn.Linear(75, 9)
        )
        self.architecture = 'conv'

    def forward(self, x):
        x = self.seq_conv(x)
        return self.seq_lin(x.flatten())


class C4ConvNetwork(nn.Module):
    def __init__(self):
        super(C4ConvNetwork, self).__init__()
        self.seq_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(4, 4), stride=(1, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(1, 1), padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=(1, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.seq_conv(x)


class C4PolicyNetwork(nn.Module):
    def __init__(self):
        super(C4PolicyNetwork, self).__init__()
        self.conv_network = C4ConvNetwork()
        self.seq_policy = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Flatten(0, -1),
            nn.Linear(1152, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 7),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv_network(x)
        return self.seq_policy(x)


class C4QNetwork(nn.Module):
    def __init__(self):
        super(C4QNetwork, self).__init__()
        self.conv_network = C4ConvNetwork()
        self.seq_value = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=3, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(),
            nn.Flatten(0, -1),
            nn.Linear(108, 32),
            # nn.BatchNorm1d(108),
            nn.LeakyReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.conv_network(x)
        return self.seq_value(x)
