import torch
import torch.nn as nn
import torch.nn.functional as F
from breakthrough import N, TOTAL_MOVES

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, padding=padding, bias=False)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        res = x
        x = F.relu(self.bn(self.conv1(x)))
        x = self.bn(self.conv2(x))
        x = x + res # skip connection
        x = F.relu(x)
        return x

class BreakThroughNet(nn.Module):
    def __init__(self):
        super(BreakThroughNet, self).__init__()
        self.board_height = N
        self.board_width = N
        self.action_size = TOTAL_MOVES

        self.resnet = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=5, padding=2),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            ResBlock(128, 128),
            ResBlock(128, 128),
            ResBlock(128, 128),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.AvgPool2d(kernel_size=4)
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
        # Fully connected layers for policy head
        self.fc_policy = nn.Linear(256, self.action_size)
        
        # Fully connected layers for value head
        self.fc_value1 = nn.Linear(256, 128)
        self.fc_value2 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = self.resnet(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Policy head (outputs action logits)
        policy_logits = self.fc_policy(x)
        
        # Value head (outputs scalar value between -1 and 1)
        value = F.relu(self.dropout(self.fc_value1(x)))
        value = torch.tanh(self.fc_value2(value))
        
        return policy_logits, value
