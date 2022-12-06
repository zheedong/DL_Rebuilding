from torchvision import models
from torch import nn

resnet18_pretrained = models.resnet18(pretrained=True,progress=True)

# Input
inp_conv = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=5)

# Output
num_ftrs = resnet18_pretrained.fc.in_features
num_classes = 10
resnet18_pretrained.fc = nn.Linear(num_ftrs, num_classes)

resnet18_pretrained = nn.Sequential(inp_conv, resnet18_pretrained)