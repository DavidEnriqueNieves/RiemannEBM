import torch
import torch.nn as nn

class Curve_Net(nn.Module):
    def __init__(self, num_channel):
        super(Curve_Net, self).__init__()
        self.num_channel = num_channel
        self.net = nn.Sequential(
            nn.Linear(64 * 3, self.num_channel),
            nn.ELU(),
            nn.Linear(self.num_channel, self.num_channel),
            nn.ELU(),
            nn.Linear(self.num_channel, self.num_channel),
            nn.ELU(),
            nn.Linear(self.num_channel, self.num_channel),
            nn.ELU(),
            nn.Linear(self.num_channel, self.num_channel),
            nn.ELU(),
            nn.Linear(self.num_channel, 64)
        )

        # self.net.apply(init_weights)

    def forward(self, x0, xT, t):
        x_cat = torch.cat([x0, xT, t], dim=1)
        return self.net(x_cat)