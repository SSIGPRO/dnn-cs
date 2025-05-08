import torch
import torch.nn as nn
import torch.nn.functional as F

class AED(nn.Module):

    def __init__(self, n, m, levels=16, sigma_soft=1.0, sigma_hard=1e7, min=-1.0, max=1.0):
        super(AED, self).__init__()
        self.n = n
        self.m = m

        self.levels = levels
        self.sigma_soft = sigma_soft
        self.sigma_hard = sigma_hard
        self.min = min
        self.max = max
        
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
        y = self.quantize(y)
        output = self.decode(y)
        label = self.discriminate(y)
        return output, label
    
    def quantize(self, inputs):
        # Subtract mean
        mean = inputs.mean(dim=-1, keepdim=True)
        inputs = inputs - mean

        # Apply tanh to squash inputs to [-1, 1]
        inputs = torch.tanh(inputs)
        
        # Create centroids
        centroids = torch.linspace(self.min, self.max, steps=self.levels, device=inputs.device)  # (levels,)
        
        # Expand dimensions to broadcast
        inputs_exp = inputs.unsqueeze(-1)                # (batch, ..., 1)
        centroids_exp = centroids.view(*([1] * inputs.ndim), self.levels)  # (1, ..., levels)

        # Compute distances
        dist = (inputs_exp - centroids_exp) ** 2         # (batch, ..., levels)

        # Soft quantization
        soft_weights = torch.softmax(-self.sigma_soft * dist, dim=-1)     # (batch, ..., levels)
        soft_quant = (soft_weights * centroids_exp).sum(dim=-1)      # (batch, ...)

        # Hard quantization
        hard_indices = dist.argmin(dim=-1)                          # (batch, ...)
        hard_one_hot = F.one_hot(hard_indices, num_classes=self.levels).type_as(dist)  # (batch, ..., levels)
        hard_quant = (hard_one_hot * centroids_exp).sum(dim=-1)     # (batch, ...)

        # Straight-through estimator
        quantized = (hard_quant - soft_quant).detach() + soft_quant

        # Add the original mean back
        quantized = quantized + mean
        return quantized
