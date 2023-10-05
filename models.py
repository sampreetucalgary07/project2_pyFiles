import torch.nn as nn
import torch


class simpleFCN(nn.Module):
    def __init__(self, first_dim, last_dim):
        super().__init__()
        self.divide_2 = int(first_dim / 2)
        self.multiply2 = int(first_dim + 2)
        self.sub2 = int(first_dim - 2)
        self.linear_model = nn.Sequential(
            nn.Linear(first_dim,  self.sub2),
            nn.ReLU(),
            nn.Linear(self.sub2, self.divide_2),
            nn.ReLU(),
            nn.Linear(self.divide_2, last_dim)
        )

    def forward(self, x1, x2):
        combined = torch.cat((x1, x2), dim=1)
        # print(combined.size())
        flat_image = self.linear_model(combined)
        return flat_image
