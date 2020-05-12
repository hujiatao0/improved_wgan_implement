import torch
from torch import nn, optim, autograd


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.net1 = nn.Sequential(
            nn.Linear(105, 1024),  # ->(32, 1024)
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 4608),  # ->(32, 4608)
            nn.BatchNorm1d(4608),
            nn.ReLU()
        )

        self.net2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),  # ->(32, 64, 12, 12)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=5, stride=2, padding=2, output_padding=1),  # ->(32, 1, 24, 24)
            nn.Tanh()
        )

    def forward(self, z):
        out = self.net1(z)  # ->(32, 4608)
        y = out.view(32, -1, 6, 6)  # ->(32, 128, 6, 6)
        out = self.net2(y)  # ->(32, 1, 24, 24)
        return out


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(6, 64, 5, 2, 2),  # ->(32, 64, 12, 12)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 5, 2, 2),  # ->(32, 128, 6, 6)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )

        self.net2 = nn.Sequential(
            nn.Linear(6 * 6 * 128, 128),  # ->(32, 128)
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 10),  # ->(32, 10)

            nn.BatchNorm1d(10),
            nn.LeakyReLU(0.2),
            nn.Linear(10, 1)  # ->(32, 1)
            # nn.Sigmoid()
        )

    def forward(self, x, y):
        z = y.view(32, 5, 1, 1).cuda()
        d = torch.ones(32, 5, 24, 24).cuda()
        zx = z * d
        z = torch.cat([x, zx], dim=1)  # ->(32, 6, 24, 24)
        # print('z shape', z.shape)
        # z = torch.tensor(z, dtype=torch.float)
        out = self.net(z)  # ->(32, 128, 6, 6)
        out = out.view(-1, 6 * 6 * 128)  # ->(32, 6 * 6 * 128)
        out = self.net2(out)
        return out


def gradient_penalty(D, xr, xf, yr):
    t = torch.rand((32, 1, 1, 1), dtype=torch.float).cuda()
    t = t.expand_as(xr)
    # t = 0.5

    mid = t * xr + (1 - t) * xf
    mid.requires_grad_()

    mid = torch.randn((32, 1, 24, 24), requires_grad=True)
    # print(mid.shape)
    pred = D(mid, yr)
    grads = autograd.grad(
        outputs=pred,
        inputs=mid,
        grad_outputs=torch.ones_like(pred),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=True
    )[0]
    # print(grads)
    grads = grads.view(32, -1)
    gp = torch.pow(grads.norm(2, dim=1) - 1, 2).mean()

    return gp

