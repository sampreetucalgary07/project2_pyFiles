import torch.nn as nn
import torch
import pickle


def model_save(model_list, patch_list, path):
    """saving models in the model_list and patch_list as pickle file in the path given"""
    for p, (model_key, patch_key) in enumerate(zip(model_list, patch_list)):
        torch.save(model_list[model_key], path+'model_'+str(p))
        """ saving patch_list which is a dictionary as a pickle file"""
    with open(path+'patch_list'+'.pkl', 'wb') as f:
        pickle.dump(patch_list, f)
    print('Models saved successfully in the path: ', path)
    print('No. of models saved: ', len(model_list))
    return None


class simpleFCN(nn.Module):
    """simple fully connected neural network with 3 hidden layers"""

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
