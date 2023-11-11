import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader

from tqdm import tqdm


class VAE(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        # Encoder ---------------------
        self.enc = nn.Sequential(
            nn.Linear(input_dim, 2**6),
            nn.ReLU(),
            nn.Linear(2**6, 2**5),
            nn.ReLU(),
            nn.Linear(2**5, 2**4),
            nn.ReLU())
        self.mu = nn.Linear(2**4, 2**3)
        self.logvar = nn.Linear(2**4, 2**3)
        # Decoder ---------------------
        self.dec = nn.Sequential(
            nn.Linear(2**3, 2**4),
            nn.ReLU(),
            nn.Linear(2**4, 2**5),
            nn.ReLU(),
            nn.Linear(2**5, 2**6),
            nn.ReLU(),
            nn.Linear(2**6, input_dim))

    def encode(self, x):
        x = self.enc(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        z = self.dec(z)
        x = torch.sigmoid(z)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    #
    # def forward_no_grad(self, input):
    #     with torch.no_grad():
    #         x_ = input.flatten()
    #         decoded, mu, logvar = self(x_)
    #     return decoded

    @classmethod
    def test__simple_call(cls, train_dataset, device=torch.device("cpu"), batch_size=64):
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)

        input_dim = torch.prod(torch.tensor(train_dataset.x[0].size())).item()
        ae = cls(input_dim=input_dim).to(device)
        x_rec = None
        for batch_idx, x, in enumerate(train_loader):
            x = x.reshape(x.shape[0], -1)
            mu = ae.mu(ae.enc(x))
            x_rec = ae.decode(mu)
            break
        return x_rec

    @classmethod
    def train_routine(cls,
            train_dataset,
            device=torch.device("cpu"),
            seed=123,
            lr=0.001,
            batch_size=64,
            n_epoch=300,
            w_loss_reg=0.1):

        torch.manual_seed(seed)

        input_dim = torch.prod(torch.tensor(train_dataset.x[0].size())).item()
        vae = VAE(input_dim=input_dim).to(device)

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)

        optimizer_vae = optim.Adam(vae.parameters(), lr=lr)

        hist_loss = []

        with tqdm(total=n_epoch, leave=True) as pbar:

            for epoch in range(n_epoch):

                for batch_idx, x, in enumerate(train_loader):

                    optimizer_vae.zero_grad()

                    x = x.reshape(x.shape[0], -1)
                    recon_x, mu, logvar = vae(x)
                    loss_rec = F.mse_loss(recon_x, x)
                    loss_reg = - torch.mean(
                        0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1),
                        dim=0)
                    loss = loss_rec + w_loss_reg * loss_reg

                    loss.backward()
                    optimizer_vae.step()

                # if epoch > 0 and epoch % 50 == 0:
                hist_loss.append(loss.item())
                pbar.update()
                pbar.set_postfix(loss=f"{loss.item():.2f}")

        vae.hist_loss = hist_loss
        return vae
