import torch
from torch import nn
from torchvision import models


class PersonalityNet(nn.Module):
    def __init__(self, num_classes):
        super(PersonalityNet, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return torch.sigmoid(self.model(x))
