import torch
import torch.nn as nn
import torch.nn.functional as F

class NNdecoder(nn.Module):

    def __init__(self, n, m):
        super(NNdecoder, self).__init__()
        self.n = n
        self.m = m
        self.fc1 = nn.Linear(self.m, 2*self.n)
        self.fc2 = nn.Linear(2*self.n, 2*self.n)
        self.fc3 = nn.Linear(2*self.n, self.n)
        self.fc4 = nn.Linear(self.n, self.n)

    def forward(self, y):
        
        z = F.relu(self.fc1(y))
        z = F.relu(self.fc2(z))
        z = F.relu(self.fc3(z))
        output = self.fc4(z) # Final layer (no activation by default)
        return output