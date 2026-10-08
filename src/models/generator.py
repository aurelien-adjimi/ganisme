from torch import nn


class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()

        self.project = nn.Linear(latent_dim, 128 * 4 * 4)

        self.layers = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.project(z)
        x = x.reshape(z.shape[0], 128, 4, 4)
        return self.layers(x)