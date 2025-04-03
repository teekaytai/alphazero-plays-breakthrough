import torch.nn as nn
import torch.nn.functional as F
from breakthrough import N, TOTAL_MOVES

class BreakThroughResNet(nn.Module):
    def __init__(self):
        super(BreakThroughResNet, self).__init__()
        self.board_height = N
        self.board_width = N
        self.action_size = TOTAL_MOVES
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Residual blocks with increasing channels
        self.res_block1 = self._make_layer(64, 64, num_blocks=3)
        self.res_block2 = self._make_layer(64, 128, num_blocks=4)
        self.res_block3 = self._make_layer(128, 256, num_blocks=6)
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(256, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * self.board_height * self.board_width, self.action_size)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(256, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * self.board_height * self.board_width, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )
    
    def _make_layer(self, in_channels, out_channels, num_blocks):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride=1, downsample=True))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.conv_block(x)
        
        # Residual blocks with increasing channels
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        
        # Policy and value heads
        policy = self.policy_head(x)
        value = self.value_head(x)
        
        return policy, value
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if downsample or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual
        out = F.relu(out)
        
        return out
    