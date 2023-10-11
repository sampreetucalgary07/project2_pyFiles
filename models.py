import torch.nn as nn
import torch
import pickle
import os


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


class simpleFCN_single(nn.Module):
    """simple fully connected neural network with 3 hidden layers"""

    def __init__(self, first_dim, last_dim):
        super().__init__()
        self.divide_2 = int(first_dim / 2)
        self.multiply2 = int(first_dim * 2)
        self.add2 = int(first_dim + 2)
        self.sub2 = int(first_dim - 2)
        self.linear_model = nn.Sequential(
            nn.Linear(first_dim,  self.add2),
            nn.ReLU(),
            nn.Linear(self.add2, self.add2),
            nn.ReLU(),
            nn.Linear(self.add2, last_dim)
        )

    def forward(self, x1):
        return self.linear_model(x1)


def model_save(model_list, patch_list, path, override=False):
    """saving models in the model_list and patch_list as pickle file in the path given"""
    if not os.path.exists(path):
        os.makedirs(path)
    elif override == True:
        pass
    else:
        ValueError('Path already exists, select override = True ')

    for p, model_key in enumerate(model_list):
        torch.save(model_list[model_key], path+'model_'+str(p+1))
        """ saving patch_list which is a dictionary as a pickle file"""
    with open(path+'patch_list'+'.pkl', 'wb') as f:
        pickle.dump(patch_list, f)
    print('Models saved successfully in the path: ', path)
    print('No. of models saved: ', len(model_list))
    return None


def model_load(path):
    """ This function will load all the 16 models and the patch_list.pkl file from the given path"""
    model_list = {}
    model_list_len = len(os.listdir(path)) - 1
    for p in range(model_list_len):
        model_list['model_' + str(p+1)] = torch.load(path+'model_'+str(p+1))
    with open(path+'patch_list.pkl', 'rb') as f:
        patch_list = pickle.load(f)

    print('Models loaded successfully from the path: ', path)
    print('No. of models loaded: ', len(model_list))
    return model_list, patch_list
