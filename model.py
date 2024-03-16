# @Author  : Peizhao Li
# @Contact : peizhaoli05@gmail.com

import torch
from torch import nn
import torch.nn.functional as F
import math


class AngularPenaltySMLoss(nn.Module):
    def __init__(self, in_features, out_features, m=4):
        super(AngularPenaltySMLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m  # margin term
        self.fc = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, labels):
        # Normalize feature vectors and weights
        with torch.no_grad():
            self.fc.weight.div_(torch.norm(self.fc.weight, dim=1, keepdim=True))
        x = F.normalize(x, p=2, dim=1)

        # Get the dot product between all feature vectors and their corresponding weights
        wf = self.fc(x)
        target_logits = wf.gather(1, labels.view(-1, 1))

        # Calculate cos(theta)
        cos_theta = target_logits
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m  # cos(m * theta)
        cos_theta_m = torch.where(cos_theta > self.th, cos_theta_m, cos_theta - self.mm)

        # One hot encode y
        one_hot = torch.zeros_like(wf)
        one_hot.scatter_(1, labels.view(-1, 1), 1)

        # Apply the new targets
        output = (one_hot * cos_theta_m) + ((1.0 - one_hot) * wf)
        loss = F.cross_entropy(output, labels)

        return loss

    def _psi(self, theta):
        m = self.m
        k = ((m * theta) / math.pi).floor()
        n_one = k % 2 * (-2) + 1  # (-1)^k
        phi_theta = ((-1) ** k) * torch.cos(m * theta) - 2 * k
        return phi_theta



class SphereCNN(nn.Module):
    def __init__(self, class_num: int, feature=False):
        super(SphereCNN, self).__init__()
        self.class_num = class_num
        self.feature = feature

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)

        self.fc5 = nn.Linear(512 * 6 * 6, 512)
        self.angular = AngularPenaltySMLoss(512, self.class_num, m=4)

    def forward(self, x, y = None):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))  # batch_size (0) * out_channels (1) * height (2) * width (3)

        x = x.view(x.size(0), -1)  # batch_size (0) * (out_channels * height * width)
        x = self.fc5(x)

        if self.feature or y is None:
            return x
        else:
            x_angle = self.angular(x, y)
            return x, x_angle


if __name__ == "__main__":
    net = SphereCNN(50)
    input = torch.ones(64, 3, 96, 96)
    output = net(input, None)