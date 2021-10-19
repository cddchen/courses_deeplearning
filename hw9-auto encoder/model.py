import torch
import torch.nn as nn


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()

        self.encoder = nn.Sequential(
            # [3, 32, 32]
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            # [64, 16, 16]
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            # [128, 8, 8]
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2)
            # [256, 4, 4] -> latent shape: 4096
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


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            # in[32, 32, 3]
            # out = (in + padding*2 - kernel_size + 1) / stride
            nn.Conv2d(in_channels=3,
                      out_channels=8,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            # size[32, 32, 8]
            nn.Conv2d(8, 16, 3, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # size[16, 16, 16]
            nn.Conv2d(16, 16, 3, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # size[8, 8, 16]
            nn.Conv2d(16, 16, 3, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # size[4, 4, 16]
            nn.Conv2d(16, 16, 3, 2, 1),
            nn.ReLU(),
            # size[2, 2, 16]
            nn.Conv2d(16, 16, 3, 2, 1),
            nn.Softmax()
            # size[1, 1, 16]
        )
        self.decoder = nn.Sequential(
            # in[1, 1, 16]
            nn.ConvTranspose2d(in_channels=16,
                               out_channels=16,
                               kernel_size=3,
                               stride=1,
                               padding=1),
            nn.ReLU(),
            # size[1, 1, 16]
            nn.ConvTranspose2d(16, 16, 3, 2, 1, output_padding=1),
            nn.ReLU(),
            # size[2, 2, 16]
            nn.ConvTranspose2d(16, 16, 3, 2, 1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # size[4, 4, 16]
            nn.ConvTranspose2d(16, 8, 3, 2, 1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            # size[8, 8, 8]
            nn.ConvTranspose2d(8, 4, 3, 2, 1, output_padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            # size[16, 16, 4]
            nn.ConvTranspose2d(4, 3, 3, 2, 1, output_padding=1),
            nn.Softmax()
            # size[32, 32, 3]
        )

    def forward(self, input: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        # input[batch_size, channel, width, height]
        out = self.encoder(input)
        # print(out.shape)
        hidden = out.reshape(out.size()[0], -1)
        # print(hidden.shape)
        out = self.decoder(out)
        # print(out.shape)
        return out, hidden