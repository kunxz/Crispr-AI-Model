# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

sequence_length = 23  

class CRISPRCNN(nn.Module):
    def __init__(self, sequence_length):
        super(CRISPRCNN, self).__init__()
        self.sequence_length = sequence_length
        #Define convolutional layers
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        #Define fully connected layers
        self.fc1 = nn.Linear(32 * sequence_length, 64)
        self.fc2 = nn.Linear(64, 1)  # Regression output

    def forward(self, x):
        #x shape: [batch_size, sequence_length, num_features]
        x = x.permute(0, 2, 1)  # Change shape to [batch_size, num_features, sequence_length]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze()
