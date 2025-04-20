import torch
import torch.nn as nn
import torch.nn.functional as F
from breakthrough import N, TOTAL_MOVES

class BreakThroughNet(nn.Module):
    def __init__(self):
        super(BreakThroughNet, self).__init__()
        self.board_height = N
        self.board_width = N
        self.action_size = TOTAL_MOVES
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

        # Fully connected layers for policy head
        self.fc_policy = nn.Linear(256 * self.board_height * self.board_width, self.action_size)

        # Fully connected layers for value head
        self.fc_value1 = nn.Linear(256 * self.board_height * self.board_width, 128)
        self.fc_value2 = nn.Linear(128, 1)

    def forward(self, x):
        # Convolutional layers with ReLU activation and batch normalization
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        # Apply dropout
        x = self.dropout(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Policy head (outputs action logits)
        policy_logits = self.fc_policy(x)

        # Value head (outputs scalar value between -1 and 1)
        value = F.relu(self.fc_value1(x))
        value = torch.tanh(self.fc_value2(value))
        return policy_logits, value

