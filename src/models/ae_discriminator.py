import torch
import torch.nn as nn
import torch.nn.functional as F

class AED(nn.Module):

    def __init__(self, n, m):
        super(AED, self).__init__()
        self.n = n
        self.m = m
        # trainable encoding matrix
        self.fc0 = nn.Linear(self.n, self.m, bias=False)
        # decoder
        self.fc1 = nn.Linear(self.m, 2*self.n)
        self.fc2 = nn.Linear(2*self.n, 2*self.n)
        self.fc3 = nn.Linear(2*self.n, self.n)
        self.fc4 = nn.Linear(self.n, self.n)
        # discriminator
        self.fc5 = nn.Linear(self.m, 2*self.n)
        self.fc6 = nn.Linear(2*self.n, 2*self.n)
        self.fc7 = nn.Linear(2*self.n, self.n)
        self.fc8 = nn.Linear(self.n, 1)

    def encode(self, x):
        return self.fc0(x)
    
    def decode(self, y):
        z = F.relu(self.fc1(y))
        z = F.relu(self.fc2(z))
        z = F.relu(self.fc3(z))
        return self.fc4(z) # Final layer (no activation by default)
    
    def discriminate(self, y):
        z = F.relu(self.fc5(y))
        z = F.relu(self.fc6(z))
        z = F.relu(self.fc7(z))
        z = self.fc8(z)
        return torch.sigmoid(z)

    def forward(self, x):
        y = self.encode(x)
        output = self.decode(y)
        label = self.discriminate(y)
        return output, label