import torch
import torch.nn as nn
import torch.nn.functional as F

class MockModel(nn.Module):
    def __init__(self):
        super(MockModel, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 20, 1, 1)
        self.conv2 = nn.Conv2d(20, 50, 1, 1)
        # Linear layers
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
        # Batch normalization
        self.batch_norm = nn.BatchNorm2d(20)
        # Embedding layer
        self.embedding = nn.Embedding(10, 10)

    def forward(self, x):
        # Convolutional layers with ReLU and max pooling
        x = F.relu(self.batch_norm(self.conv1(x)))
        x = F.relu(self.conv2(x))
        # Flatten the tensor
        x = x.view(-1, 4*4*50)
        # Linear layers with ReLU
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
