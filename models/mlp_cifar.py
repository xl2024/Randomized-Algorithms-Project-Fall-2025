import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, num_classes=10, hidden_dim=128, depth=4):
        super(MLP, self).__init__()
        
        input_dim = 32*32*3
        
        layers = []
        
        # Input Layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden Layers
        for _ in range(depth - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            
        # Output Layer
        layers.append(nn.Linear(hidden_dim, num_classes))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # Flatten: [Batch, 3, 32, 32] -> [Batch, 3072]
        x = x.view(x.size(0), -1)
        return self.net(x)

def MLP_CIFAR(num_classes=10):
    return MLP(num_classes=num_classes)