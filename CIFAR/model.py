import torch.nn as nn


class BasicBlockNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=16, kernel_size=3, padding=1
            ),  # (16, 32, 32)
            nn.MaxPool2d(2),  # (16, 16, 16)
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, padding=1
            ),  # (32, 16, 16)
            nn.MaxPool2d(2),  # (32, 8, 8)
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, padding=1
            ),  # (64, 8, 8)
            nn.MaxPool2d(2),  # (64, 4, 4)
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, padding=1
            ),  # (128, 4, 4)
            nn.ReLU(),
        )

        self.classifier = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        feature_map = self.net(x)  # (B, 128, 4, 4)
        feature_vector = feature_map.mean(dim=(2, 3))  # (B, 128)
        logits = self.classifier(feature_vector)  # (B, 10)
        return logits
