import torch as th
from torch import nn


class BorjaCNN(nn.Module):
    def __init__(self, input_shape):
        output_shape = 1  # 1 number is needed: the amount of reward.
        super().__init__()
        n_input_channels = input_shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=7, stride=3, padding=0),
            nn.BatchNorm2d(16),
            nn.Dropout2d(p=0.2),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm2d(16),
            nn.Dropout2d(p=0.2),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.Dropout2d(p=0.2),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.Dropout2d(p=0.2),
            nn.LeakyReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            x = th.rand(input_shape)
            n_flatten = self.cnn(th.as_tensor(x[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, output_shape), nn.ReLU())

    def forward(self, x):
        # adds gaussian noise
        with th.no_grad():
            x += th.normal(0, 0.1, size=x.shape, device=x.device)
        return self.linear(self.cnn(x))
