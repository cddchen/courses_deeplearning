import torch
import torch.nn as nn


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()

        self.encoder = nn.Sequential(
            # [3, 32, 32]
            nn.Conv2d(3, 64, 3, stride=1, dilation=1),
            nn.ReLU(True),
            # [64, 30, 30]
            nn.Conv2d(64, 128, 3, stride=1, dilation=2),
            nn.ReLU(True),
            # [128, 26, 26]
            nn.Conv2d(128, 256, 3, stride=1, dilation=5),
            nn.ReLU(True),
            # [256, 16, 16]
            nn.Conv2d(256, 256, 3, stride=1, dilation=1),
            nn.ReLU(True),
            # [256, 14, 14]
            nn.Conv2d(256, 256, 3, stride=1, dilation=5),
            nn.ReLU(True),
            # [256, 4, 4]
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, stride=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 9, stride=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 17, stride=1),
            nn.Tanh()
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x = self.decoder(x1)
        return x1, x
