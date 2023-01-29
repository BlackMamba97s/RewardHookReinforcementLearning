import torch
import torch.nn as nn
import torch.optim as optim
import torchvision


class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.feature_extractor = torchvision.models.resnet50(pretrained=True)
        self.fc_project = nn.Linear(in_features=1000, out_features=2048)
        self.fc1 = nn.Linear(in_features=2048, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=2)
        self.log_std = nn.Parameter(torch.zeros(1, 2))


    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.fc_project(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def sample(self, x):
        x = self.feature_extractor(x)
        x = self.fc_project(x)
        x = self.fc1(x)
        mean = self.fc2(x)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob


class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.feature_extractor = torchvision.models.resnet50(pretrained=True)
        self.fc1 = nn.Linear(in_features=1000, out_features=2048) # added this
        self.fc2 = nn.Linear(in_features=2048, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=1)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.fc1(x) # changed this
        x = self.fc2(x)
        x = self.fc3(x)
        return x