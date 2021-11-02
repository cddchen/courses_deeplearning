import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(32*32*3, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 32*32*3)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparamerize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparamerize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_VAE(recon_x, x, mu, log_var, criterion):
    mse = criterion(recon_x, x)
    # loss=0.5*sum(1+log(sigma^2)-mu^2-sigma^2)
    KLD_element = mu.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return mse + KLD


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        # conv out size:
        # out = (in+2*padding-dilation*(kernel_size-1)-1)/stride+1
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

        # ConvTranspose2d out size:
        # out = (in-1)*stride-2*padding+dilation*(kernel_size-1)+output_padding+1
        self.decoder = nn.Sequential(
            # [256, 4, 4]
            nn.ConvTranspose2d(256, 128, 5, stride=1),
            # [128, 8, 8]
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 9, stride=1),
            # [64, 16, 16]
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 17, stride=1),
            # [3, 32, 32]
            nn.Tanh()
            # [-1, 1]
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x = self.decoder(x1)
        return x1, x