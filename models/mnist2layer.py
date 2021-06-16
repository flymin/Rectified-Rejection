import torch.nn as nn
import torch.nn.functional as F
# normalize: 0.13, 0.31
# original input size: 28 x 28


class Mnist2LayerNet(nn.Module):
    def __init__(self, num_classes=10, use_BN=False, out_dim=10, along=False):
        super(Mnist2LayerNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5, stride=1)
        self.linear = nn.Sequential(
            nn.Linear(4 * 4 * 50, 500),
            nn.ReLU(),
            nn.Linear(500, 10))
        if use_BN:
            self.dense = nn.Sequential(
                nn.Linear(4 * 4 * 50, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, out_dim)
            )
            print('with BN')
        else:
            self.dense = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, out_dim)
            )
        self.along = along

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        feature = x.view(-1, 4 * 4 * 50)
        classification_return = self.linear(feature)
        if self.along:
            evidence_return = self.dense(feature)
        else:
            evidence_return = self.linear(feature) + self.dense(feature)
        return classification_return, evidence_return
