import torch.nn as nn
import torch


class simpleFCN(nn.Module):
    def __init__(self, first_dim, last_dim):
        super().__init__()
        # self.flatten = nn.Flatten()
        self.divide_2 = int(first_dim / 2)
        # print(self.divide_2)
        self.multiply2 = int(first_dim * 2)
        # print(self.multiply2)
        self.linear_model = nn.Sequential(
            nn.Linear(first_dim, self.multiply2),
            nn.ReLU(),
            nn.Linear(self.multiply2, first_dim),
            nn.ReLU(),
            nn.Linear(first_dim, last_dim)
        )

    def forward(self, x1, x2):
        combined = torch.cat((x1, x2), dim=1)
        # print(combined.size())
        flat_image = self.linear_model(combined)
        return flat_image
