import torch
from torch import nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(2,2)
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(24)
        self.fc1 = nn.Linear(2904, 10)

    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))      
        output = self.pool(output)                        
        output = F.relu(self.bn4(self.conv4(output)))     
        output = output.view(-1, 2904)
        output = self.fc1(output)

        return output