########################################################################
# Define a Convolution Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Multiple fully connected layers. 

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, relu = True):
        super(Net, self).__init__()
        self.relu = relu
        self.fc1 = nn.Linear(3 * 32 * 32, 120)  #fully connected
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        if self.relu:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        else:
            x = self.fc1(x)
            x = self.fc2(x)
        x = self.fc3(x)
        return x
