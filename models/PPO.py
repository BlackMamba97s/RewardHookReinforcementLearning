import torch
import torch.nn as nn
import torch.optim as optim
import torchvision


class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        # Use a pre-trained ResNet-50 model as the feature extractor
        self.feature_extractor = torchvision.models.resnet50(pretrained=True)
        # Linear layer to project features from 1000 to 2048
        self.fc_project = nn.Linear(in_features=1000, out_features=2048)
        # Linear layers to reduce features to 512 and 2
        self.fc1 = nn.Linear(in_features=2048, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=2)
        # Parameter to store the standard deviation of the normal distribution
        self.log_std = nn.Parameter(torch.zeros(1, 2))

    def forward(self, x):
        # Use the feature extractor to extract features from the input
        x = self.feature_extractor(x)
        # Pass the features through the linear layers
        x = self.fc_project(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def sample(self, x):
        # Use the feature extractor to extract features from the input
        x = self.feature_extractor(x)
        x = self.fc_project(x)
        x = self.fc1(x)
        # Compute the mean of the normal distribution
        mean = self.fc2(x)
        # Compute the standard deviation of the normal distribution
        std = torch.exp(self.log_std)
        # Create a normal distribution
        dist = torch.distributions.Normal(mean, std)
        # Sample an action from the distribution
        action = dist.sample()
        # Compute the log probability of the action
        log_prob = dist.log_prob(action)
        return action, log_prob


class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        # Use a pre-trained ResNet-50 model as the feature extractor
        self.feature_extractor = torchvision.models.resnet50(pretrained=True)
        # Linear layers to reduce features to 2048, 512, and 1
        self.fc1 = nn.Linear(in_features=1000, out_features=2048)
        self.fc2 = nn.Linear(in_features=2048, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=1)

    def forward(self, x):
        # Use the feature extractor to extract features from the input
        x = self.feature_extractor(x)
        # Pass the features through the linear layers
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x